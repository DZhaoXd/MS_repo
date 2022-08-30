"""
在以下文件的当中存在保存和使用案例，可参考使用以下案例。
提供测试版本的保存多模型脚本，若有异常问题，可联系相关人员进行反馈。
"""

### 保存部分案例
import mindspore

"""
案例一：

# param_list用于整合多个模型的参数，不同模型的参数通过name_prefix添加前缀来区别，并且加载后用于划分。
# 根据需求此部分进行修改。
param_list = [{'name': name, 'data': param} for name, param in
              model.parameters_and_names(name_prefix='net_G')]
for name, param in model_D1.parameters_and_names(name_prefix='net_D1'):
    param_list.append({'name': name, 'data': param})
for name, param in model_D2.parameters_and_names(name_prefix='net_D2'):
    param_list.append({'name': name, 'data': param})

# append_dict 用于保存中间的各项参数情况，官方介绍仅值类型支持int，float，bool。
append_dict = {'iter': int(optimizer.global_step.asnumpy()[0]),
               'mIoU': float(miou),
               'best_IoU': float(best_iou) if miou < best_iou else float(miou)}
mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)


案例二：使用自定义的保存多模型函数进行保存。

model = DeeplabMulti(num_classes=19)
model_D1 = FCDiscriminator(num_classes=19)
model_D2 = FCDiscriminator(num_classes=19)

save_path = './checkpoint/temp.ckpt'
model_dict = {'net_G': model,
              'net_D1': model_D1,
              'net_D2': model_D2}

append_dict = {'iter': 10, 'lr': 0.01, 'Acc': 0.98}

model_save_multi(save_path=save_path,models_dict=model_dict,append_dict=append_dict)
state = mindspore.load_checkpoint(save_path)
state_dict = split_checkpoint(state,model_dict.keys())
print(state_dict.keys())

"""


### 加载划分函数

def split_checkpoint(checkpoint, split_list=None):
    """
    Input：
    checkpoint:待划分的模型参数
    split_list:待划分的模型参数前缀名
    """
    if split_list == None:
        return checkpoint
    checkpoint_dict = {name: {} for name in split_list}
    for key, value in checkpoint.items():
        prefix = key.split('.')[0]
        if prefix not in checkpoint_dict:
            checkpoint_dict[key] = value.asnumpy()
            continue
        name = key.replace(prefix + '.', '')
        checkpoint_dict[prefix][name] = value
    return checkpoint_dict


def model_save_multi(save_path: str,
                     models_dict: dict,
                     append_dict=None, print_save=True) -> None:
    """
    Input:
    save_path:模型保存的路径
    models_dict:多模型字典，例如：{'net_G': model,'net_D1': model_D1,'net_D2': model_D2}
    append_dict:附加信息字典，例如：{'iter': 10, 'lr': 0.01, 'Acc': 0.98}
    print_save:是否打印模型保存路径
    Output：
    None
    """
    params_list = []
    for model_name, model in models_dict.items():
        for name, param in model.parameters_and_names(name_prefix=model_name):
            params_list.append({'name': name, 'data': param})
    mindspore.save_checkpoint(params_list, save_path, append_dict=append_dict)
    if print_save:
        print('Save success , The model save path : {}'.format(save_path))


"""
### 断点续训-使用案例

split_list = ['net_G', 'net_D1', 'net_D2']
train_state_dict = mindspore.load_checkpoint(args.continue_train)
train_state_dict = split_checkpoint(train_state_dict, split_list=split_list)
iter_start = train_state_dict['iter']
best_iou = train_state_dict['best_IoU']
mindspore.load_param_into_net(model, train_state_dict['net_G'])
mindspore.load_param_into_net(model_D1, train_state_dict['net_D1'])
mindspore.load_param_into_net(model_D2, train_state_dict['net_D2'])
optimizer.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
optimizer_D1.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
optimizer_D2.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))

"""
