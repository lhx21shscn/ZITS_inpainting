import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datasets.dataset_teacher import *
from src.models.Teacher_model import *
from .inpainting_metrics import get_inpainting_metrics
from .utils import Progbar, create_dir, stitch_images, SampleEdgeLineLogits


class LaMa_Teacher:
    def __init__(self, config, gpu, rank, test=False):
        self.config = config
        self.device = gpu
        self.global_rank = rank

        self.model_name = 'teacher'

        kwargs = dict(config.training_model)
        kwargs.pop('kind')

        self.inpaint_model = TeacherInpaintingTrainingModule(config, gpu=gpu, rank=rank, test=test, **kwargs).to(gpu)

        if config.min_sigma is None:
            min_sigma = 2.0
        else:
            min_sigma = config.min_sigma
        if config.max_sigma is None:
            max_sigma = 2.5
        else:
            max_sigma = config.max_sigma

        self.train_dataset = AuxDataset(config.TRAIN_FLIST, mask_path=config.TRAIN_MASK_FLIST,
                                        batch_size=config.BATCH_SIZE // config.world_size, augment=True, training=True,
                                        test_mask_path=None, train_line_path=config.train_line_path,
                                        add_pos=config.use_MPE, world_size=config.world_size,
                                        min_sigma=min_sigma, max_sigma=max_sigma)
        if config.DDP:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config.world_size,
                                                    rank=self.global_rank, shuffle=True)
        self.val_dataset = AuxDataset(config.VAL_FLIST, mask_path=None,
                                      batch_size=config.BATCH_SIZE, augment=False, training=False,
                                      test_mask_path=config.TEST_MASK_FLIST,
                                      eval_line_path=config.eval_line_path,
                                      add_pos=config.use_MPE, input_size=config.INPUT_SIZE,
                                      min_sigma=min_sigma, max_sigma=max_sigma)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.val_path = os.path.join(config.PATH, 'validation')
        create_dir(self.val_path)

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

        self.best = float("inf") if self.inpaint_model.best is None else self.inpaint_model.best

    def save(self):
        if self.global_rank == 0:
            self.inpaint_model.save()

    def train(self):
        if self.config.DDP:
            train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE // self.config.world_size,
                                      num_workers=12, sampler=self.train_sampler)
        else:
            train_loader = DataLoader(self.train_dataset, pin_memory=True,
                                      batch_size=self.config.BATCH_SIZE, num_workers=12, shuffle=True)

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.config.world_size

        if total == 0 and self.global_rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:
            epoch += 1
            if self.config.DDP:
                self.train_sampler.set_epoch(epoch + 1)  # Shuffle each epoch
            epoch_start = time.time()
            if self.global_rank == 0:
                print('\n\nTraining epoch: %d' % epoch)
            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter', 'loss_scale'],
                              verbose=1 if self.global_rank == 0 else 0)

            for _, items in enumerate(train_loader):
                self.inpaint_model.train()

                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)

                # train
                outputs, gen_loss, dis_loss, logs, batch = self.inpaint_model.process(items)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break
                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + [(i, logs[0][i]) for i in logs[0]] + [(i, logs[1][i]) for i in logs[1]]
                if self.config.No_Bar:
                    pass
                else:
                    progbar.add(len(items['image']),
                                values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 1 and self.global_rank == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 1 and self.global_rank == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 1:
                    if self.global_rank == 0:
                        print('\nstart eval...\n')
                        print("Epoch: %d" % epoch)
                    psnr, ssim, fid = self.eval()
                    if self.best > fid and self.global_rank == 0:
                        self.best = fid
                        print("current best epoch is %d" % epoch)
                        print('\nsaving %s...\n' % self.inpaint_model.name)
                        raw_model = self.inpaint_model.generator.module if \
                            hasattr(self.inpaint_model.generator, "module") else self.inpaint_model.generator
                        torch.save({
                            'iteration': self.inpaint_model.iteration,
                            'generator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_gen.pth'))
                        raw_model = self.inpaint_model.discriminator.module if \
                            hasattr(self.inpaint_model.discriminator, "module") else self.inpaint_model.discriminator
                        torch.save({
                            'discriminator': raw_model.state_dict(),
                            'best_fid': fid,
                            'ssim': ssim,
                            'psnr': psnr
                        }, os.path.join(self.config.PATH, self.inpaint_model.name + '_best_dis.pth'))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 1 and self.global_rank == 0:
                    self.save()
            if self.global_rank == 0:
                print("Epoch: %d, time for one epoch: %d seconds" % (epoch, time.time() - epoch_start))
                logs = [('Epoch', epoch), ('time', time.time() - epoch_start)]
                self.log(logs)
        print('\nEnd training....')

    def eval(self):
        if self.config.DDP:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE // self.config.world_size,  ## BS of each GPU
                                    num_workers=12)
        else:
            val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True,
                                    batch_size=self.config.BATCH_SIZE, num_workers=12)

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        if self.config.No_Bar:
            pass
        else:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        with torch.no_grad():
            for items in tqdm(val_loader):
                iteration += 1
                for k in items:
                    if type(items[k]) is torch.Tensor:
                        items[k] = items[k].to(self.device)
                b, _, _, _ = items['image'].size()

                # inpaint model
                # eval
                items = self.inpaint_model(items)
                outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))
                # save
                outputs_merged *= 255.0
                outputs_merged = outputs_merged.permute(0, 2, 3, 1).int().cpu().numpy()
                for img_num in range(b):
                    cv2.imwrite(self.val_path + '/' + items['name'][img_num], outputs_merged[img_num, :, :, ::-1])

        our_metric = get_inpainting_metrics(self.val_path, self.config.GT_Val_FOLDER, None, fid_test=True)

        if self.global_rank == 0:
            print("iter: %d, PSNR: %f, SSIM: %f, FID: %f, LPIPS: %f" %
                  (self.inpaint_model.iteration, float(our_metric['psnr']), float(our_metric['ssim']),
                   float(our_metric['fid']), float(our_metric['lpips'])))
            logs = [('iter', self.inpaint_model.iteration), ('PSNR', float(our_metric['psnr'])),
                    ('SSIM', float(our_metric['ssim'])), ('FID', float(our_metric['fid'])), ('LPIPS', float(our_metric['lpips']))]
            self.log(logs)
        return float(our_metric['psnr']), float(our_metric['ssim']), float(our_metric['fid'])

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()
        with torch.no_grad():
            items = next(self.sample_iterator)
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].to(self.device)
            # inpaint model
            iteration = self.inpaint_model.iteration
            inputs = (items['image'] * (1 - items['mask']))
            if self.config.Edge:
                inputs_edge = items['edge'] * (1 - items['mask'])
            if self.config.Line:
                inputs_line = items['line'] * (1 - items['mask'])
            if self.config.Seg:
                inputs_seg = items['seg'] * (1 - items['mask'])
            items = self.inpaint_model(items)
            outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        if self.config.Line:
            images = stitch_images(
                self.postprocess(items['image'].cpu()),
                self.postprocess(inputs.cpu()),
                self.postprocess(inputs_line.cpu()),
                self.postprocess(items['mask'].cpu()),
                self.postprocess(items['predicted_image'].cpu()),
                self.postprocess(items['predicted_line'].cpu()),
                self.postprocess(outputs_merged.cpu()),
                img_per_row=image_per_row
            )
        else:
            images = stitch_images(
                self.postprocess(items['image'].cpu()),
                self.postprocess(inputs.cpu()),
                self.postprocess(items['mask'].cpu()),
                self.postprocess(items['predicted_image'].cpu()),
                self.postprocess(outputs_merged.cpu()),
                img_per_row=image_per_row
            )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[0]) + '\t' + str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
