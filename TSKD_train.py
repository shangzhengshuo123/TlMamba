import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import torch.nn.functional as F
import timm
from models.TlMamba import VSSM as tlmamba
from torch.autograd import Variable
from torchvision import datasets
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 定义训练和验证函数
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch, model_ema):
    model.train()
    # AverageMeter()用来管理一些需要更新的变量，包括loss，ACC1，ACC5
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    # total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    # 训练过程可视化
    train_bar = tqdm(train_loader, file=sys.stdout)
    for batch_idx, (data, target) in enumerate(train_bar):
        data, target = data.to(device, non_blocking=True), Variable(target).to(device,non_blocking=True)
        # 将数据输入mixup_fn生成mixup数据
        samples, targets = mixup_fn(data, target)
        # 将第三部生成的mixup数据输入model，输出预测结果，然后再计算loss 向前传播
        # output = model(samples)
        # optimizer.zero_grad()梯度清零，把loss关于weight的导数变成0
        optimizer.zero_grad()

        # 教师模型预测 (不计算梯度)
        with torch.no_grad():
            teacher_output = teacher_model(samples)

        # 学生模型预测
        student_output = model(samples)

        # 如果使用混合精度
        if use_amp:
            # with torch.cuda.amp.autocast()开启混合精度
            with torch.cuda.amp.autocast():
                # 原始分类损失
                cls_loss = torch.nan_to_num(criterion_train(student_output, targets))
                # 确保教师和学生输出维度一致
                assert teacher_output.shape == student_output.shape, \
                    f"Teacher output shape {teacher_output.shape} != student output shape {student_output.shape}"
                # 蒸馏损失 (使用KL散度)
                temperature = 4.0  # 可以尝试2.0-4.0之间的值  原2.0
                distill_loss = F.kl_div(
                    F.log_softmax(student_output / temperature, dim=1),
                    F.softmax(teacher_output / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)  # 温度参数T=2.0
                # 组合损失 (可调整权重)
                loss = cls_loss * 0.7 + distill_loss * 0.3
                # # 计算loss.torch.nan_to_num将输入中的NaN、正无穷大和负无穷大替换为NaN、posinf和neginf。默认情况下，nan会被替换为零，正无穷大会被替换为输入的dtype所能表示的最大有限值，负无穷大会被替换为输入的dtype所能表示的最小有限值。
                # loss = torch.nan_to_num(criterion_train(output, targets))
            # 梯度放大
            scaler.scale(loss).backward()
            # 梯度裁剪，放置梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            # Unscales gradients and calls
            # or skips optimizer.step()
            # 首先把梯度值unscale回来，如果梯度值不是inf或NaN,则调用optimizer.step()来更新权重，否则，忽略step调用，从而保证权重不更新
            scaler.step(optimizer)
            # Updates the scale for next iteration
            # 更新下一次迭代的scaler
            scaler.update()
        # 否则，直接反向传播求梯度。torch.nn.utils.clip_grad_norm_函数执行梯度裁剪，防止梯度爆炸。
        else:
            # loss = criterion_train(output, targets)
            # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(models.parameters(), CLIP_GRAD)
            # optimizer.step()

            # 非混合精度情况
            cls_loss = criterion_train(student_output, targets)
            distill_loss = F.kl_div(
                F.log_softmax(student_output / 2.0, dim=1),
                F.softmax(teacher_output / 2.0, dim=1),
                reduction='batchmean'
            ) * (2.0 * 2.0)
            loss = cls_loss * 0.6 + distill_loss * 0.4  #原0.3
            if batch_idx % 50 == 0:
                print(
                    f"[Distill Monitor] Epoch {epoch} | Batch {batch_idx} | cls_loss: {cls_loss.item():.4f} | distill_loss: {distill_loss.item():.4f}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        # 如果use_ema为True，则执行model_ema的updata函数，更新模型
        # model_ema是一个利用当前模型创建出的滑动平均模型。这个模型可以用于在训练过程中更新参数，以提高模型的性能。在训练结束后，可以使用model_ema进行测试或验证，以评估模型的性能。
        if model_ema is not None:
            model_ema.update(model)
        # 等待上面所有的操作执行完成
        torch.cuda.synchronize()
        # 自定义调整学习率的方法
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # 更新loss，ACC1,ACC5的值
        loss_meter.update(loss.item(), target.size(0))
        # acc1是只有真实标签是预测出的概率最高的类，才算预测正确  acc5是只有真实标签是预测出的概率最高的5个类之一，就算预测正确
        acc1, acc5 = accuracy(student_output, target, topk=(1, 5))
        # loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    # 等待一个epoch训练完成后，计算平均loss和平均acc
    ave_loss =loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc


# 验证过程
# 在val的函数上面添加@torch.no_grad()，作用：所有计算得出的tensor的requires_grad都自动设置为False。即使一个tensor（命名为x）的requires_grad = True，在with torch.no_grad计算，由x得到的新tensor（命名为w-标量）requires_grad也为False，且grad_fn也为None,即不会对w求导
@torch.no_grad()
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    # 测试的loss
    loss_meter = AverageMeter()
    # top1的ACC
    acc1_meter = AverageMeter()
    # top5的ACC
    acc5_meter = AverageMeter()
    # 总的验证集的数量
    # total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    # 验证集的label
    val_list = []
    # 预测的label
    pred_list = []
    # 训练过程可视化
    test_bar = tqdm(test_loader, file=sys.stdout)

    for data, target in test_bar:
        # 将label保存到val_list
        for t in target:
            val_list.append(t.data.item())
        # 将data和target放入device上，non_blocking设置为True
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        # 将data输入到model中，求出预测值，然后输入到loss函数中，求出loss
        # print('data====', data.shape)
        output = model(data)
        loss = criterion_val(output, target)
        # 调用torch.max函数，将预测值转为对应的label
        _, pred = torch.max(output.data, 1)
        # 将输出的预测值的label存入pred_list
        for p in pred:
            pred_list.append(p.data.item())
        # 调用accuracy函数计算ACC1和ACC5
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # 更新loss_meter、acc1_meter、acc5_meter的参数
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    # 本次epoch循环完成后，求得本次epoch的acc、loss
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg,  acc,  acc5_meter.avg))
    # 保存模型的逻辑
    # 如果ACC比Best_ACC高，则保存best模型
    # 判断模型是否为DP方式训练的模型
    # 如果是DP方式训练的模型，模型参数放在model.module，则需要保存model.module。
    # 否则直接保存model。
    # 注：保存best模型，我们采用保存模型权重信息。
    if acc > Best_ACC:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), file_dir + '/' + 'best.pth')
        else:
            torch.save(model.state_dict(), file_dir + '/' + 'best.pth')
        Best_ACC = acc
    # 接下来保存每个epoch的模型。
    # 判断模型是否为DP方式训练的模型。
    # 如果是DP方式训练的模型，模型参数放在model.module，则需要保存model.module.state_dict()。
    # 新建个字典，放置Best_ACC、epoch和 model.module.state_dict()等参数。然后将这个字典保存。判断是否是使用EMA，如果使用，则还需要保存一份ema的权重。
    # 否则，新建个字典，放置Best_ACC、epoch和 model.state_dict()等参数。然后将这个字典保存。判断是否是使用EMA，如果使用，则还需要保存一份ema的权重。

    # 注意：对于每个epoch的模型只保存了state_dict参数，没有保存整个模型文件
    if isinstance(model, torch.nn.DataParallel):
        state = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema'] = model.module.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    else:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema'] = model.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
        # print('22222222222')
    return val_list, pred_list, loss_meter.avg, acc

@torch.no_grad()
def eval_teacher(model, device, test_loader):
    model.eval()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    val_list, pred_list = [], []

    print("\nEvaluating Teacher Model...")
    for data, target in tqdm(test_loader, desc="Teacher Eval"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output.data, 1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        val_list.extend(target.cpu().tolist())
        pred_list.extend(pred.cpu().tolist())

    print("\n Teacher Model Accuracy:")
    print("Top-1 Accuracy: {:.2f}%".format(acc1_meter.avg))
    print("Top-5 Accuracy: {:.2f}%".format(acc5_meter.avg))
    print("\n Classification Report:")
    print(classification_report(val_list, pred_list))


# 设置了固定的随机因子，再次训练的时候就可以保证图片的加载顺序不会发生变化
def seed_everything(seed=42):
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 创建保存模型的文件夹
    file_dir = ''
    # 设置存放权重文件的文件夹，如果文件夹存在删除再建立
    if os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    else:
        os.makedirs(file_dir)

    # 设置全局参数
    model_lr = 1e-4  # 学习率
    BATCH_SIZE = 32  #16
    EPOCHS = 300
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # 是否使用混合精度
    use_dp = True  # 是否开启dp方式的多卡训练
    classes = 51  # 类别个数（肯定要修改）
    resume = None  # 再次训练的模型路径，如果不为None,则表示加载resume指向的模型继续训练
    CLIP_GRAD = 5.0  # 梯度的最大范数，在梯度裁剪里设置
    Best_ACC = 0  # 记录最高ACC得分
    use_ema = True  # 是否使用ema
    model_ema_decay = 0.9998
    start_epoch = 1  # 开始的epoch，默认是1，如果重新训练时，需要给start_epoch重新赋值
    seed = 1  # 随机因子，数值可以随意设定，但是设置后，不要随意更改，更改后，图片加载的顺序会改变，影响测试结果
    seed_everything(seed)

    # 数据预处理和数据增强 加入了随机10度的旋转、高斯模糊、色彩饱和度明亮度的变化、Mixup等比较常用的增强手段，做了Resize和归一化
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        # transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ([0.90062904, 0.90062904, 0.90062904], [0.26650605, 0.26650605, 0.26650605])
        transforms.Normalize(mean=[0.90062904, 0.90062904, 0.90062904], std= [0.26650605, 0.26650605, 0.26650605])

    ])
    transform_test = transforms.Compose([
        # 由于选用的FastViT模型输入是256*256的大小，所以Resize为256*256
        # transforms.Resize((64, 64)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 设置为计算mean和std
        transforms.Normalize(mean=[0.90062904, 0.90062904, 0.90062904], std= [0.26650605, 0.26650605, 0.26650605])
    ])
    # Mixup：数据增强
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=classes)

    # 读取数据
    dataset_train = datasets.ImageFolder('train', transform=transform)
    dataset_test = datasets.ImageFolder("test", transform=transform_test)
    with open('xidaiclass.txt', 'w') as file:
        file.write(str(dataset_train.class_to_idx))
    with open('xidaiclass.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据 rop_last设置为True，因为使用了Mixup数据增强，必须保证每个batch里面的图片个数为偶数（不能为零），如果最后一个batch里面的图片为奇数，则会报错，所以舍弃最后batch的迭代
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    # 设置Loss 训练的loss为：SoftTargetCrossEntropy，验证的loss：nn.CrossEntropyLoss()
    # 实例化模型并且移动到GPU
    criterion_train = SoftTargetCrossEntropy()
    criterion_val = torch.nn.CrossEntropyLoss()

    # 设置模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 学生模型（TlMamba）
    model_ft = tlmamba(num_classes=classes).to(device)
    num_fr = model_ft.head.in_features
    model_ft.head = nn.Linear(num_fr, classes)
    print('model_ft.head====', model_ft.head)

    # 教师模型（预训练的 FastViT）
    # teacher_model = deit_tiny_patch16_224(pretrained=False)
    teacher_model = timm.create_model('pit_b_224', num_classes=51, pretrained=True).to(device)
    num_features = teacher_model.head.in_features
    teacher_model.head = nn.Linear(num_features, classes)
    teacher_model.load_state_dict(
        torch.load(''))

    # 冻结所有参数（teacher 只用于推理）
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model.eval()
    teacher_model.to(device)

    # 获取head的in_features，然后，指定head的输出数量为classes
    num_fr = model_ft.head.in_features
    model_ft.head = nn.Linear(num_fr, classes)
    print('model_ft.head====', model_ft.head)
    # 如果resume设置为已经训练的模型的路径，则加载模型接着resume指向的模型接着训练，使用模型里的Best_ACC初始化Best_ACC，使用epoch参数初始化start_epoch
    if resume:
        model = torch.load(resume)
        print(model['state_dict'].keys())
        model_ft.load_state_dict(model['state_dict'])
        Best_ACC = model['Best_ACC']
        start_epoch = model['epoch']+1
        # print('666666666666666666666')
    model_ft.to(DEVICE)
    # 如果模型输出是classes的长度，则表示修改正确了
    # print(model_ft)

    # 设置优化器和学习率调整策略
    # 选择简单暴力的Adam优化器，学习率调低 优化器设置为adamW，学习率调整策略选择为余弦退火
    optimizer = optim.AdamW(model_ft.parameters(),lr=model_lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    # print('555555555555555555555')
    # 设置混合精度，DP多卡，EMA
    # use_amp为True，则开启混合精度训练,声明pytorch自带的混合精度 torch.cuda.amp.GradScaler()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    # 检测可用显卡的数量，如果大于1，并且开启多卡训练的情况下，则要用torch.nn.DataParallel加载模型，开启多卡训练
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = torch.nn.DataParallel(model_ft)
    # 如果使用ema，则注册ema
    if use_ema:
        model_ema = ModelEma(
            model_ft,
            decay=model_ema_decay,
            device=DEVICE,
            resume=resume)
    else:
        model_ema = None

    # 调用训练和验证方法
    # 训练与验证

    # 定义参数
    # 是否已经设置了学习率，当epoch大于一定的次数后，会将学习率设置到一定的值，并将其置为True
    is_set_lr = False
    # 记录log用的，将有用的信息保存到字典中，然后转为json保存起来
    log_dir = {}
    # train_loss_list保存每个epoch的训练loss val_loss_list保存每个epoch的验证loss train_acc_list保存每个epoch的训练acc val_acc_list保存每个epoch的验证acc epoch_list存放每个epoch的值
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    # print('55555555555555555555555555')
    # 如果是接着上次的断点继续训练则读取log文件，然后把log取出来，赋值到对应的list上
    if resume and os.path.isfile(file_dir+"result.json"):
        with open(file_dir+'result.json', 'r', encoding='utf-8') as file:
            logs = json.load(file)
            train_acc_list = logs['train_acc']
            train_loss_list = logs['train_loss']
            val_acc_list = logs['val_acc']
            val_loss_list = logs['val_loss']
            epoch_list = logs['epoch_list']
    # print('44444444444444444444444')
    # 循环epoch
    for epoch in range(start_epoch, EPOCHS + 1):
        # print('33333333333333333333')
        epoch_list.append(epoch)
        log_dir['epoch_list'] = epoch_list
        # 评估 teacher 模型（可选）
        # eval_teacher(teacher_model, DEVICE, test_loader)
        # 调用train函数，得到 train_loss, train_acc，并将分别放入train_loss_list，train_acc_list，然后存入到logdir字典中。
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch, model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # print('111111111111111111111111111111')
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        # 调用验证函数，判断是否使用EMA？
        # 如果使用EMA，则传入model_ema.ema，否则，传入model_ft。
        # 得到val_list, pred_list, val_loss, val_acc。将val_loss, val_acc分别放入val_loss_list和val_acc_list中，然后存入到logdir字典中。
        # print('model_ema.ema=====', model_ema.ema)
        if use_ema:
            # print('test_loader=========', test_loader)
            # print('model_ema.ema=====', model_ema.ema)
            val_list, pred_list, val_loss, val_acc = val(model_ema.ema, DEVICE, test_loader)
        else:
            val_list, pred_list, val_loss, val_acc = val(model_ft, DEVICE, test_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        # 保存log
        with open(file_dir + './result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        # 打印本次的测试报告
        # print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))
        # 如果epoch大于600，将学习率设置为固定的1e-6
        if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True
        # 绘制loss曲线和acc曲线
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)
