import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

from utils.logger import *


def summary(model, input_size, batch_size=-1, device="cuda", single=True):
    def register_hook(module):

        def hook(module, input, output):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # print(input[0].shape)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if isinstance(input[0], list):
                summary[m_key]["input_shape"] = list(list(i.size()) for i in input[0])
            else:
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    # if isinstance(input_size, tuple):
    # input_size = [input_size]

    log.info(input_size)
    # batch_size of 2 for batch norm
    x = [torch.rand(batch_size, *in_size).type(dtype) for in_size in input_size]
    # log.info(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass

    # single input or list input
    if single:
        model(x)
    else:
        model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    log.info("----------------------------------------------------------------")
    line_new = "{:<20} {:<40}{:<40} {:<15}".format("Layer (type)", "input Shape", "Output Shape", "Param #")
    log.info(line_new)
    log.info("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<20} {:<40}{:<40} {:<15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        if isinstance(summary[layer]["output_shape"][0], list):
            for lout in summary[layer]["output_shape"]:
                total_output += np.prod(lout)
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        log.info(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    log.info("================================================================")
    log.info("Total params: {0:,}".format(total_params))
    log.info("Trainable params: {0:,}".format(trainable_params))
    log.info("Non-trainable params: {0:,}".format(total_params - trainable_params))
    log.info("----------------------------------------------------------------")
    log.info("Input size (MB): %0.2f" % total_input_size)
    log.info("Forward/backward pass size (MB): %0.2f" % total_output_size)
    log.info("Params size (MB): %0.2f" % total_params_size)
    log.info("Estimated Total Size (MB): %0.2f" % total_size)
    log.info("----------------------------------------------------------------")
    # return summary
