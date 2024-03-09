<h1><font face="黑体" size=10 color=white>2024.2.26 Nougat/train.py</font></h1>
<font face="宋体" size=5 color=white>
<ul>
    <li>主要部分是`train`函数，这个函数只有一个参数，`cfg`，所有内容定义在`cfg`中，把`cfg`传给`train`就可以直接训练，值得学习
    </li>
    <li>`train`中先设置随机种子，然后定义模型和数据集，再定义各种回调函数，包括学习率回调、checkpoint回调、自定义的梯度监测回调。然后就是logger，接着就是使用lightning.pytorch中的训练框架，进行定义及训练
    </li>
    <li>细节部分就是各种自定义的回调函数如何定义以及各种接口如何使用的。这个是需要好好研究的部分。
    </li>
</ul>
</font>

<h1><font face="黑体" size=10 color=white>2024.2.27 Nougat/test.py</font></h1>
<font face="宋体" size=5 color=white>
<ul>
    <li>`test.py`中只有一个函数test，入口是args，返回模型预测的结果。</li>
    <li>先定义了模型，导入checkpoint，将模型设置为`eval`模式，接着make test结果的文件夹，然后准备数据集，dataloader，然后就是for循环里依次将每个batch喂到模型里，将产生的结果送到get_metric函数计算指标，保存指标，返回预测结果。</li>
    <li>里面有个多线程并行的内容值的学习一下，Pool池去做的，写在csdn中了，但是我实验出来并行花的时间更多不知道什么情况。</li>
</ul>
</font>


<h1><font face="黑体" size=10 color=white>2024.2.28 Nougat/predict.py</font></h1>
<font face="宋体" size=5 color=white>
<ul>
    <li>`predict.py`先导入模型，将模型设置成eval模式，再准备dataset，准备dataloader，dataset是每个pdf用一个LazyDataset进行定义，然后append到datasets里，用ConcatDataset合成一个。然后进行inference，然后对模型结果进行处理保存。</li>
</ul>
</font>


<h1><font face="黑体" size=10 color=white>2024.2.29 Nougat/lightning_module.py</font></h1>
<font face="宋体" size=5 color=white>
<ul>
    <li>这个文件非常重要，定义了使用pytorch ligntning训练所需的Model Module和Data Module</li>
    <li>`NougatModelPLModule`类继承自`pl.LightningModule`，初始化中定义了模型。<br>
    接下来是内置函数`training_step`的重写，定义了模型训练时的行为，做法就是把batch的数据取出来，通过模型获得loss，最后也返回loss。
    <br>
    然后是`validation_step`的重写，读取batch中的数据，使用model进行inference，得到pred的文本和原来的groundtruth文本，然后计算metrics，最后返回metrics。
    <br>
    定义了个钩子函数`on_validation_epoch_end`,在validation epoch的最后调用，内部的操作就是使用了log_dict记录了一下metrics，这个log_dict是框架自己定义的一个函数，还没了解透。
    <br>
    `configure_optimizers`函数是训练优化器的所在，画一大段程序算了个没用到的max_iter???不是很懂。然后定义了optimizer使用的是torch现有的库，参数是模型参数，用的是self.parameters()我也没搞懂，不应该是self.model.parameters()吗，还有个参数是lr，比较简单。接着定义了schedule，模式化的东西，是一个字典，里面包含一些关键字，“scheduler”是一个函数，这个函数也是有范式可循，就是学习率怎么调整，指数还是余弦等等。最后返回两个东西，一个是optimizer，一个是schedule，两个都分别放到了一个列表里。
    <br>
    然后就是怎么去写scheduler了，scheduler返回一个LambdaLR的函数，可以去代码里看看是怎么写的，就是定义了一个名为`lr_lambda`的函数用来计算学习率。
    <br>
    `NougatDataPLModule`类继承自`pl.LightningDataModule`，初始化中定义了数据读取。里面有两个改写后的函数`train_dataloader`和`val_dataloader`里面分别有一个loaders的list，至于为什么写我还不知道，先follow吧，list里是一个torch带的DataLoader。
    </li>
</ul>
</font>


<h1><font face="黑体" size=10 color=white>2024.3.1 Nougat/总结</font></h1>
<font face="宋体" size=5 color=white>
&emsp;All right! 我们来总结一下nougat的外面的代码，包括train，test，predict还有定义module的代码，形成大局观，以便下次我们再编写使用pytorch lightning框架的时候能有一个清晰的思路。
<ul>
    <li>首先看下train的代码，首先要有config定义模型的各种参数，然后定义模型，定义数据集，主要就是这两个，还有就是一些辅助的函数了，一些callback，一些log的内容。</li>
    <ul>
        <li>具体到代码内部，就nougat这个项目而言，首先初始化seed，定义model module，data module。model module并没有具体的操作，data module是对train和val的数据进行了定义，使用了`NouagtDataset`对`data_module.train_datasets`和`data_module.val_datasets`进行了替换，需要注意的是，`data_module.train_datasets`和`data_module.val_datasets`本身是一个列表，列表里面的元素是个`NouagtDataset`类的实例，这例子中只有一个这样的`NougatDasteset`，结合`lighning_module.py`里的代码来看,数据module里面的`train_loader`和`val_loader`导入的是一个`ConcatDataset`类，估计是支持把多个Dataset合起来的操作。</li>
        <li>定义完了module，就是一些对模型监视的callbacks，这些callback有的需要自定义，有的直接调用`pytorch.lightning`内部写好的类进行实例化，自己写的callback要继承自`lightning.pytorh.callbacks.Callback`，然后内部定义钩子函数指明白啥时候调用，比如`on_after_backward`在反向传播之后调用。</li>
        <li>callback之后是log，log是要传到Trainer里的，通常情况下使用`TensorBoardLogger`，参数的话一个是保存路径，一个是这个logger叫的name，还有version，其他参数要去查阅官网教程。</li>
        <li>最后就是定义`Trainer`了，trainer有很多参数，但是具体需要查阅官网，`Trainer`与model module和data moudle无关主要定义训练相关的参数。`trainer.fit`与model module和data moudle有关，其他参数参考官网相关教程。</li>
        <li>最后值的说明的一点，所有配置都在config中，可以使用`sconf.Config`对config.yaml文件进行管理，另外可以通过argparse开放一些参数接口给用户使用，`trainer.py`处理的很好。</li>
    </ul>
    <li>`test.py`比`train.py`简单。对于test来说，最主要的就是训练好的模型在测试集上的指标，跟lightning这个框架关系不大。首先是导入预训练好的模型，模型放到想用的accelerator上去，设置成<font color=red><b>eval</b></font>模式，定义好数据集和dataloader，for循环将每个batch的数据喂给模型，然后获得结果，计算指标，将计算好的指标存起来。</li>
    <li>`predict.py`用来完成实际任务，跟test的思路差不多，用不到lightning框架。导入模型和数据跑就完事，将结果变成我们需要的，比如nougat将模型的prediction变成markdown格式。</li>
    <li>重要文件：`lightning_moduel.py`定义了lightning格式的model module和data module。
    <ul>
        <li>首先是`NougatModelPLModule`继承自`pl.LightningModule`，`__init__()`函数里首先继承父类的`__init__()`，接着定义需要的一些内部属性，并定义了<font color=red><b>self.model</b></font>，接着就是重写两个框架内的函数`training_step`和`validation_step`，注意两者的入口参数都要有`batch`和`batch_id`，还可以有`dataset_idx`。`training_step`中做的就是用模型跑数据，返回一个loss，并且对结果进行log。`validation_step`中做的也是跑数据，与train不同，与test相同的是，validation需要返回各种的metric，也进行了结果的log。</li>
        <li>重写完了训练和验证，另一个重要的重写函数是配置优化器的函数`configure_optimizers`，这个函数没有其他参数，返回两个列表，前一个列表中是optimizer，后一个列表中是scheduler，看样子每个列表里面可以配置多个。optimizer的配置直接使用`torch.optim`中定义的优化器即可，scheduler的配置是一个字典，里面会有几个固定的key的内容要填，“scheduler”，“name”，“interval”，“frequency”等，其他参数可以参考lightning的官网。其中“scheduler”是一个LambdaLR对象可以自己去编写，有格式可循，可以去看一下代码是咋实现的参考一下。</li>
        <li>接着就是定义一些钩子函数，比如`on_validation_epoch_end`在valid epoch的最后log一下metric啥的。还可以定义一些其他lightning框架自定义的函数，用来完成包括save_checkpoint等操作，可以通过查看官网得知。</li>
        <li>`NougatDataPLModule`继承自`pl.LightningDataModule`,这个module主要定义了train和val的dataset，重写两个内部的方法`train_dataloader`和`val_dataloader`，他们分别返回一个list，list中是一个`torch`内部定义的`Dataloader`的实例化，看样子可以有多个dataloader。其他的就是一些辅助的函数了，比如dataloader中用到的初始化seed的函数，或者是collate_fn等。</li>
    </ul>
    </li>
</ul>
</font>


<h1><font face="黑体" size=10 color=white>2024.3.3 Nougat/nougat/model.py</font></h1>
<font face="宋体" size=5 color=white>
终于来到了模型部分，nougat是由两个模型拼起来的，一个是swintransform，另一个是MBartForCausalLM的decoder部分。下面讲一讲model.py的思路。
<ul>
    <li>分为四个部分，一个是`SwinEncoder`，一个是`BARTDecoder`部分，然后就是将两者结合到一起的`NougatModel`。其他部分是上面三个类中用到的辅助类，包括继承自`PretrainedConfig`的`NougatConfig`，还有用于decoder早停的类`StoppingCriteriaScores`。</li>
    <li>`SwinEncoder`继承自`nn.Module`，的初始化函数做的就是定义了`self.model`，直接通过hugging face中timm库的类`SwinTransformer`去定义的。并导入了pretrain好的ckpt，并且对不能直接导入的参数进行了修改。`forward`函数就是一套前向传播的流程完事。还有就是一些辅助函数，主要是对输入图片进行操作。</li>
    <li>`BARTDecoder`继承自`nn.Module`，初始化中首先进行了参数的初始化，然后对tokenizer进行了config，直接调用现成的模型`MBartForCausalLM`定义`self.model`，对model进行config，包括了导入pretrained model并对新模型中不适应的dict进行处理。`forward`中直接调用`self.model.forward`进行了前向传播。其他方法主要是一些辅助方法，position embedding weight进行规整等（变成合适于max_length的长度）</li>
    <li>`NougatModel`继承自`PreTrainedModel`，初始化中传入一个config的类，此处是继承自`PretrainedConfig`的类`NougatConfig`，然后直接调用上面定义的两个定义好的encoder和decoder作为`self.encoder`和`self.decoder`。`forward`就是传入数据前向传播。`inference`做法整体上是数据导到模型产出结果，然后对结果进行各种处理，判断有没有重复等等。值得注意的一个点是其中对各种参数的设置，比如其中的decoder中的`stopping_criteria`，决定什么时候停止继续decode，停止decode的时候其实就是出现重复的时候，文件中定义了终止的类，继承自`StoppingCriteria`。作者应该是通过实验得出了出现重复的规律进行规避。</li>
    
</ul>
</font>

<h1><font face="黑体" size=10 color=white>2024.3.4 Nougat/nougat/transforms.py</font></h1>
<font face="宋体" size=5 color=white>
自定义的数据增强
<ul>
    <li>`transorms.py`中使用了流行的`albumentations`库对图片进行增强，整体来说就是定义transform给model程序去调用。</li>
</ul>
</font>

<h1><font face="黑体" size=10 color=white>2024.3.4 Nougat/nougat/metrics.py</font></h1>
<font face="宋体" size=5 color=white>
`metrics.py`中使用`nltk`包计算包括`edit_distance`，`bleu`，`meteor`，`precision`，'recall`，`f_measure`在内的各种指标。其中最主要的两个函数是`compute_metrics`以及`get_metrics`，前者用于计算各种指标，后者用于并行化计算多组metrics。其他的函数都是些辅助函数，使输入符合指标计算函数。

</font>


<h1><font face="黑体" size=10 color=white>2024.3.8 Nougat/utils/dataset.py</font></h1>
<font face="宋体" size=5 color=white>
构建用于训练的数据集，没有特别的地方，主要是熟悉构架数据集的流程。首先自定义的数据集继承自`torch.utils.data.Dataset`，然后至少有三个方法，`__init__`用于初始化，`__len__(self)`，返回dataset的长度，`__getitem__(self, index)`用于从数据集中取数据，返回用于训练的数据。还可以定义些辅助函数，里面定义的静态方法`ignore_none_collate(batch)`不知道咋去用的，好像也不是钩子函数，里面调用了`torch.utils.data.dataloader.default_collate`函数，将batch中的数据叠起来。见Jay E的CSDN博客。
</font>

<h1><font face="黑体" size=10 color=white>2024.3.8 Nougat/utils/device.py+checkpoint.py</font></h1>
<font face="宋体" size=5 color=white>
<ul>
    <li>`device.py`中根据cuda中的空间大小确定了batch size的大小，他这里的batch size设成GB单位的显存数量的0.3倍我有点不理解。还有一个函数用来将模型量化成bf16，以及搬到特定的device上去，值的学习。</li>
    <li>`checkpoint.py`用来下载模型，其中的一个下载函数非常值的学习，通过request请求，使用流的方式，并且带进度条，bravo。具体做法是，先得到一个请求结果resp，从其中的header中得到总长度，resp参数中的streamm=True表示使用流的方式，即不是一下子把所用内容都取出来，而是一步一步的，先提取header，再在需要的时候取数据。这个函数要背下来，明天写到自己的utils里面。其他的就是一些辅助的函数了，获取路径啦，执行具体文件的下载啦等等。</li>
</ul>
</font>