from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd

from ..utils.distributed import tensor_gather


class OIM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,
                lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets,
                              lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, \
            lut, cq, header, momentum = ctx.saved_tensors

        inputs, targets = \
            tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * \
                    lut[y] + (1. - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets,
                     lut, cq,
                     torch.tensor(header), torch.tensor(momentum))


class OIMSMR(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,
                lut, cq, cq_omega,
                omega_decay, momentum):
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())

        omega_x_unlabeled = outputs_labeled.max(dim=1)[0] /  \
            outputs_unlabeled.max(dim=1)[0]

        ctx.save_for_backward(inputs, targets, omega_x_unlabeled,
                              lut, cq, cq_omega,
                              omega_decay, momentum)

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, omega_x_unlabeled, \
            lut, cq, cq_omega, \
            omega_decay, momentum = ctx.saved_tensors

        inputs, targets, omega_x_unlabeled = \
            tensor_gather((inputs, targets, omega_x_unlabeled))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))

        for x, y, omega in zip(inputs, targets, omega_x_unlabeled):
            if y < len(lut):
                lut[y] = momentum * \
                    lut[y] + (1. - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                if omega < cq_omega.min():
                    continue
                else:
                    header_cq = cq_omega.argmin().item()
                    cq[header_cq] = x
                    cq_omega[header_cq] = omega

        cq_omega *= omega_decay

        return grad_inputs, None, None, None, None, None, None


def oimsmr(inputs, targets, lut, cq, cq_omega, omega_decay=0.99, momentum=0.5):
    return OIMSMR.apply(inputs, targets,
                        lut, cq, cq_omega,
                        torch.tensor(omega_decay), torch.tensor(momentum))

class OIMLoss(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))
        self.register_buffer('cq',  torch.zeros(
            self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = (label >= 0)
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(
            inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq,
                        self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = ((self.header_cq +
                           (label >= self.num_pids).long().sum().item()) %
                          self.num_unlabeled)
        loss_oim = F.cross_entropy(projected, label,
                                   ignore_index=5554)
        return loss_oim


class OIMLossSMR(nn.Module):

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar,
                 omega_decay=0.99):
        super(OIMLossSMR, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.omega_decay = omega_decay
        self.oim_scalar = oim_scalar

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))
        self.register_buffer('cq',  torch.zeros(
            self.num_unlabeled, self.num_features))
        self.register_buffer('cq_omega', torch.zeros(
            self.num_unlabeled))

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = (label >= 0)
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(
            inputs)].view(-1, self.num_features)

        projected = oimsmr(inputs, label, self.lut, self.cq,
                           self.cq_omega, self.omega_decay, momentum=self.momentum)
        projected *= self.oim_scalar

        loss_oim = F.cross_entropy(projected, label,
                                   ignore_index=5554)
        return loss_oim
