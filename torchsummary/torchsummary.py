import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

from collections import OrderedDict
import numpy as np

from models.imagenet.mobilenetv2 import mobilenetv2, InvertedResidual
from utils import imagenet1000_clsidx_to_labels as lables

from os import path
import math


def save_q_bit_rule(model, bit_list, file_path, w_bit_width, b_bit_width):
    w_param = [
        p for p in model.state_dict()
        if ('weight' in p and not 'classifier' in p)
    ]
    b_param = [
        p for p in model.state_dict()
        if ('bias' in p and not 'classifier' in p)
    ]
    assert len(w_param) == len(bit_list['int_w_weight'])
    assert len(b_param) == len(bit_list['int_w_bias'])
    file_name = path.join(file_path, 'rule.txt')
    with open(file_name, 'w') as handler:
        for name, i in zip(w_param, bit_list['int_w_weight']):
            handler.write('.*\\.{} fixed_point {} {}\n'.format(
                name, w_bit_width, i))
        for name, i in zip(b_param, bit_list['int_w_bias']):
            handler.write('.*\\.{} fixed_point {} {}\n'.format(
                name, b_bit_width, i))
    print('Rule has been saved in {}.'.format(file_name))


def summary(model,
            image_file,
            input_size=192,
            batch_size=-1,
            device="cuda",
            w_q_bit=12,
            b_q_bit=8,
            save_data=False,
            file_path='./'):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            # module_idx = len(summary)

            if isinstance(module, InvertedResidual):
                block[0] += 1

                res = "_Rsdu_" if module.identity else ""
                exp = "_Expnd_{}".format(module.expand_ratio)
                strd = "_Strd_{}".format(module.stride)

                state = "%s%s%s%s" % (class_name, res, exp, strd)

                IR[0].add(state)

                m_key = "%s_%i" % (state, block[0])

            elif isinstance(module, nn.Conv2d):
                i_c_size = input[0].size()[1]
                i_h_size = input[0].size()[2]
                o_c_size = output.size()[1]
                o_h_size = output.size()[2]
                k_size = module.kernel_size[0]
                st_size = module.stride[0]
                p_size = module.padding[0]
                specs.append([
                    i_c_size, i_h_size, o_c_size, o_h_size, k_size, st_size,
                    p_size
                ])
                
                max_in_chan[0] = max(max_in_chan[0], i_c_size)
                max_out_size[0] = max(max_out_size[0], o_h_size)

                if module.kernel_size[0] == 3: 
                    if module.groups > 1:  # It is a DW Conv
                        conv3x3_dw_num[0] += 1
                        m_key = "IR_%i_%s(3x3)_dw_%i" % (
                            block[0] + 1, class_name, conv3x3_dw_num[0])
                        
                        acc_size = np.prod(output.size()[2:])
                    else:
                        acc_size = input[0].size()[1] * np.prod(
                            output.size()[2:])
                        conv3x3_norm_num[0] += 1
                        m_key = "%s(3x3)_%i" % (class_name,
                                                conv3x3_norm_num[0])
                elif module.kernel_size[0] == 1:
                    acc_size = input[0].size()[1]  #*np.prod(output.size()[2:])
                    conv1x1_num[0] += 1
                    # Make sure it is not the last conv.
                    if output.size()[0] != 1280:
                        m_key = "IR_%i_%s(1x1)_%i" % (block[0] + 1, class_name,
                                                      conv1x1_num[0])
                    else:
                        m_key = "%s(1x1)_%i" % (class_name, conv1x1_num[0])
            else:
                ReLU[0] += 1
                m_key = "%s_%i" % (class_name, ReLU[0])

            if save_data[0]:
                if isinstance(module, InvertedResidual) or isinstance(
                        module, nn.Conv2d) or isinstance(
                            module, nn.AdaptiveAvgPool2d):
                    np.save(path.join(file_path[0], m_key + "_output"),
                            output.detach().cpu())
                    np.save(path.join(file_path[0], m_key + "_input"),
                            input[0].detach().cpu())
                # if isinstance(module, nn.Conv2d):
                #    np.save(
                #        path.join(file_path[0], m_key+'_weights'), module.weight.detach().cpu())
                #    np.save(
                #        path.join(file_path[0], m_key+'_bias'), module.bias.detach().cpu())

            summary[m_key] = OrderedDict()

            if isinstance(module, nn.Conv2d):
                w = module.weight.detach().cpu()
                b = module.bias.detach().cpu()
                min_v = w.min()
                max_v = w.max()
                b_max_v = b.max()
                b_min_v = b.min()

                max_min_weight[0] = max(max_min_weight[0], max_v)
                max_min_weight[1] = min(max_min_weight[1], min_v)
                max_min_bias[0] = max(max_min_bias[0], b_max_v)
                max_min_bias[1] = min(max_min_bias[1], b_min_v)

                summary[m_key]["weight_range"] = [min_v.item(), max_v.item()]
                summary[m_key]["bias_range"] = [b_min_v.item(), b_max_v.item()]
                summary[m_key]["spr_w"] = (
                    1 - (float(np.count_nonzero(w)) / np.prod(w.size()))) * 100
                summary[m_key]["spr_b"] = (
                    1 - (float(np.count_nonzero(b)) / np.prod(b.size()))) * 100

                int_bit = math.ceil(math.log2(max(abs(min_v), abs(max_v))))
                b_int_bit = math.ceil(
                    math.log2(max(abs(b_min_v), abs(b_max_v))))

                if int_bit < 0:
                    int_bit = 0

                if b_int_bit < 0:
                    b_int_bit = 0

                # if min_v < 0 and int_bit == 0:
                #    int_bit = 1
                # if b_min_v < 0 and b_int_bit == 0:
                #    b_int_bit = 1

                summary[m_key]["int_bit_size"] = int_bit
                summary[m_key]["b_int_bit_size"] = b_int_bit

                int_bit_val['int_w_weight'].append(int_bit)
                int_bit_val['int_w_bias'].append(b_int_bit)

                max_acc_buf_size[0] = max(max_acc_buf_size[0], acc_size)

            input_buf_size = np.prod(input[0].size()[1:])
            output_buf_size = np.prod(output.size()[1:])

            if input_buf_size > output_buf_size:
                if max_buf_size_mb[0] < input_buf_size:
                    max_buf_size_mb[0] = input_buf_size
                    for i in range(3):
                        max_buf_size_value[i] = max(max_buf_size_value[i],
                                                    input[0].size()[i + 1])
            else:
                if max_buf_size_mb[0] < output_buf_size:
                    max_buf_size_mb[0] = output_buf_size
                    for i in range(3):
                        max_buf_size_value[i] = max(max_buf_size_value[i],
                                                    output.size()[i + 1])

            # m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["max_input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:]
                                                  for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(
                    torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(
                    module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)):
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
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    # x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))
    # NOTE: Let's open a predefined file since I want to save input and output as golden model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_it = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    image = Image.open(image_file)

    x = transform_it(image).type(dtype)
    x = x.unsqueeze(0)

    calc_confidence = torch.nn.functional.softmax

    # create properties
    summary = OrderedDict()

    max_buf_size_value = list([float('-Inf'), float('-Inf'), float('-Inf')])
    max_buf_size_mb = list([float('-Inf')])

    save_data = list([save_data])
    file_path = list([file_path])

    conv1x1_num = list([0])
    conv3x3_norm_num = list([0])
    conv3x3_dw_num = list([0])
    ReLU = list([0])
    block = list([-1])
    max_min_weight = list([float('-Inf'), float('+Inf')])
    max_min_bias = list([float('-Inf'), float('+Inf')])
    IR = list([set()])
    int_bit_val = OrderedDict()
    int_bit_val['int_w_bias'] = list()
    int_bit_val['int_w_weight'] = list()

    max_out_size = list([float('-Inf')])
    max_in_chan = list([float('-Inf')])
    max_acc_buf_size = list([float('-Inf')])

    specs = list()

    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    y = model(x)
    y = y.view(-1, 1000)

    conf = calc_confidence(y)
    top5_val, top5_idx = conf.topk(5)
    top5_idx = top5_idx.cpu().numpy()
    for idx, conf_ in zip(top5_idx[0], top5_val[0]):
        print('{}: {:2f}%'.format(lables.cls_idx[idx], conf_ * 100))

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>40}  {:>25} {:>25} {:>15} {:>15} {:>15} {:>25} {:>15} {:>15}".format(
        "Layer (type)", "Output Shape", "Weight range", "Int. bit",
        "Weight Sparsity", "Bias Sparsity", "Bias range", "Bias Int. bit",
        "Param #")
    print(line_new)
    print(
        "==========================================================================================================================="
    )
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        if 'conv' in layer.lower():
            # input_shape, output_shape, w range, int bit, trainable, nb_params
            line_new = "{:>40}  {:>25} {:>25} {:>15} {:>15} {:>15} {:>25} {:>15} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{:2f}, {:2f}".format(summary[layer]["weight_range"][0],
                                      summary[layer]["weight_range"][1]),
                str(summary[layer]["int_bit_size"]),
                "{:2f}".format(summary[layer]["spr_w"]),
                "{:2f}".format(summary[layer]["spr_b"]),
                "{:2f}, {:2f}".format(summary[layer]["bias_range"][0],
                                      summary[layer]["bias_range"][1]),
                str(summary[layer]["b_int_bit_size"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            if 'invertedresidual' in layer.lower():
                IR.append(layer)
            line_new = "{:>40}  {:>25} {:>25} {:>15} {:>15} {:>25} {:>15} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                # "{:2f}, {:2f}".format(summary[layer]["weight_range"][0], summary[layer]["weight_range"][1]),
                "-----",
                "-----",  # str(summary[layer]["int_bit_size"]),
                "-----",
                "-----",
                "-----",
                "-----",
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024**2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024**2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024**2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params -
                                               trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("Max Buf Size: %s, %i cells, %.2f(Mb) " %
          (str(max_buf_size_value), np.prod(max_buf_size_value),
           max_buf_size_mb[0] * w_q_bit / (1024 * 1024)))
    print("Weight range: %s" % (str(max_min_weight)))
    print("Bias range: %s" % (str(max_min_bias)))
    print("----------------------------------------------------------------")
    print("Saving rule for bit quantization...")
    print("IR States:")
    for state in IR[0]:
        print(state)
    save_q_bit_rule(model, int_bit_val, file_path[0], w_q_bit, b_q_bit) 

    return max_buf_size_value, specs, [max_in_chan[0], max_out_size[0], max_acc_buf_size[0]] 
