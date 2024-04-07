import os
import torch.distributed as dist
from torch.nn import SyncBatchNorm as SynBN
from torch.nn.parallel import DistributedDataParallel as DDP

from src.losses.adversarial import NonSaturatingWithR1
from src.losses.feature_matching import masked_l1_loss, feature_matching_loss
from src.losses.perceptual import ResNetPL
from src.models.LaMa import *
from src.models.TSR_model import *
from src.models.upsample import StructureUpsampling
from src.utils import get_lr_schedule_with_warmup, torch_init_model


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


class TeacherBaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('TeacherBaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.edge = config.Edge
        self.line = config.Line
        self.seg = config.Seg
        self.generator = Teacher(config.Edge, config.Line, config.Seg, \
                                 in_channels=config.generator["input_nc"]).cuda(gpu)
        self.best = None

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

        self.load()
        if self.config.DDP:
            self.generator = self.generator if not config.DDP else SynBN.convert_sync_batchnorm(self.generator)  # BN层同步
            self.generator = DDP(self.generator)
            self.discriminator = self.discriminator if not config.DDP else SynBN.convert_sync_batchnorm(self.discriminator)  # BN层同步
            self.discriminator = DDP(self.discriminator)

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])

        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
                if torch.cuda.is_available():
                    data = torch.load(gen_weights_path)
                else:
                    data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)
                self.best = data['best_fid']
                print('Loading best fid...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


class TeacherInpaintingTrainingModule(TeacherBaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        self.refine_mask_for_losses = None


    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
        input_data = [masked_img, mask]
        if self.edge:
            masked_edge = batch['edge'] * (1 - mask)
            input_data.append(masked_edge)
        if self.line:
            masked_line = batch['line'] * (1 - mask)
            input_data.append(masked_line)
        if self.seg:
            masked_seg = batch['seg'] * (1 - mask)
            input_data.append(masked_seg)
        input_data = torch.cat(input_data, dim=1)
        predicted_image, predicted_edge, predicted_line, predicted_seg, predicted_feat  = self.generator(input_data.to(torch.float32))
        batch['predicted_image'] = predicted_image
        batch['inpainted_image'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        if self.edge:
            batch['predicted_edge'] = predicted_edge
            batch['inpainted_edge'] = mask * batch['predicted_edge'] + (1 - mask) * batch['edge']

        if self.line:
            batch['predicted_line'] = predicted_line
            batch['inpainted_line'] = mask * batch['predicted_line'] + (1 - mask) * batch['line']

        if self.seg:
            batch['predicted_seg'] = predicted_seg
            batch['inpainted_seg'] = mask * batch['predicted_seg'] + (1 - mask) * batch['seg']

        batch['predicted_feat'] = predicted_feat
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1

        self.discriminator.zero_grad()
        # discriminator loss
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])

        real_loss, _, _ = self.adversarial_loss.discriminator_real_loss(real_batch=batch['image'],
                                                                  discr_real_pred=discr_real_pred)
        batch = self.forward(batch)
        predicted_img = batch[self.image_to_discriminator].detach()

        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))

        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])

        dis_loss = fake_loss + real_loss

        dis_metric = {}
        dis_metric['discr_adv'] = dis_loss.item()
        dis_metric.update(add_prefix_to_keys(dis_metric, 'adv_'))

        dis_loss.backward()
        self.dis_optimizer.step()

        # generator loss
        self.generator.zero_grad()
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        gen_loss = l1_value
        gen_metric = dict(gen_l1=l1_value.item())

        # vgg-based perceptual loss
        if self.config.losses['perceptual']['weight'] > 0:
            pl_value = self.loss_pl(predicted_img, img,
                                    mask=supervised_mask).sum() * self.config.losses['perceptual']['weight']
            gen_loss = gen_loss + pl_value
            gen_metric['gen_pl'] = pl_value.item()

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        gen_loss = gen_loss + adv_gen_loss
        gen_metric['gen_adv'] = adv_gen_loss.item()
        gen_metric.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            discr_real_pred, discr_real_features = self.discriminator(img)
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            gen_loss = gen_loss + fm_value
            gen_metric['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            gen_loss = gen_loss + resnet_pl_value
            gen_metric['gen_resnet_pl'] = resnet_pl_value.item()

        if self.edge:
            edge_loss = F.binary_cross_entropy(batch["predicted_edge"].permute(0, 2, 3, 1).contiguous().view(-1, 1), \
                                               batch["edge"].permute(0, 2, 3, 1).contiguous().view(-1, 1)) * self.config.losses['edge']['weight']
            gen_loss = gen_loss + edge_loss
            gen_metric["edge"] = edge_loss.item()

        if self.line:
            line_loss = F.binary_cross_entropy(batch["predicted_line"].permute(0, 2, 3, 1).contiguous().view(-1, 1), \
                                               batch["line"].permute(0, 2, 3, 1).contiguous().view(-1, 1)) * self.config.losses['line']['weight']
            gen_loss = gen_loss + line_loss
            gen_metric["line"] = line_loss.item()

        # TODO delete
        if self.seg:
            pass

        if self.config.AMP:
            self.scaler.scale(gen_loss).backward()
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            gen_metric['loss_scale'] = self.scaler.get_scale()
        else:
            gen_loss.backward()
            self.gen_optimizer.step()
        # create logs
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch
