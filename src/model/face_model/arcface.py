import torch
import torchvision.transforms as T
from torch import nn

from src.model.face_model.model_irse import Backbone


def un_norm_clip(x1):
    x = x1 * 1.0  # to avoid changing the original tensor or clone() can be used
    reduce = False
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        reduce = True
    x[:, 0, :, :] = x[:, 0, :, :] * 0.26862954 + 0.48145466
    x[:, 1, :, :] = x[:, 1, :, :] * 0.26130258 + 0.4578275
    x[:, 2, :, :] = x[:, 2, :, :] * 0.27577711 + 0.40821073

    if reduce:
        x = x.squeeze(0)
    return x


class Arcface(nn.Module):
    def __init__(self, path, multiscale=False):
        super(Arcface, self).__init__()
        print("Loading ResNet ArcFace")
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        # self.facenet=iresnet100(pretrained=False, fp16=False) # changed by sanoojan

        self.facenet.load_state_dict(torch.load(path))

        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        self.set_requires_grad(False)

    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def extract_feats(self, x, clip_img=True):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = T.functional.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = (
            self.face_pool_1(x) if x.shape[2] != 256 else x
        )  # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x)  # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale)

        # x_feats = self.facenet(x) # changed by sanoojan
        return x_feats

    def forward(self, y_hat, y, clip_img=True, return_seperate=False):
        n_samples = y.shape[0]
        y_feats_ms = self.extract_feats(
            y, clip_img=clip_img
        )  # Otherwise use the feature from there

        y_hat_feats_ms = self.extract_feats(y_hat, clip_img=clip_img)
        y_feats_ms = [y_f.detach() for y_f in y_feats_ms]

        loss_all = 0
        sim_improvement_all = 0
        seperate_sim = []
        for y_hat_feats, y_feats in zip(y_hat_feats_ms, y_feats_ms):
            loss = 0
            sim_improvement = 0
            count = 0
            # lossess = []
            for i in range(n_samples):
                sim_target = y_hat_feats[i].dot(y_feats[i])
                sim_views = y_feats[i].dot(y_feats[i])

                seperate_sim.append(sim_target)
                loss += 1 - sim_target  # id loss
                sim_improvement += float(sim_target) - float(sim_views)
                count += 1

            loss_all += loss / count
            sim_improvement_all += sim_improvement / count
        if return_seperate:
            return loss_all, sim_improvement_all, seperate_sim
        return loss_all, sim_improvement_all, None
