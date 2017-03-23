# TensorFlow-Intel-Interview

[TOC]

## 使用TensorFlow来进行二次函数的拟合

为了初步了解TensorFlow的使用我先根据对tensorflow最基本的了解来写一个Demo，进行函数的拟合。然后再去看看Tensorflow Serving的事情。

我在Mac和Linux上面都进行了TensorFlow的安装。在Linux下还比较正常，但是在Mac下面因为“six”这个模块的版本过低导致报了一些错。我升级了six这个模块，好像就好了。

```shell
sudo easy_install -U six
```

现在我们继续运行函数的拟合工作。

首先创建导入包。

```python
# -*- coding: utf-8 -*-
#设计一个直线拟合程序
import tensorflow as tf
import numpy as np
```

然后创造一个线性100位的随机数列矩阵作为x_data，并使用使用一个二次函数通过x_data求出对应的y_data。

```python
#首先生成一个训练集，主要就是100个点。
#生成x。这个是一个1-100的随机数列
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*x_data + 2*x_data + 3

print x_data
print y_data
```

然后我们就可以打印出这两个线性矩阵。

```she
[ 0.81370246  0.42756972  0.22368494  0.06153482  0.42600977  0.18890364
  0.37369776  0.34966856  0.67164725  0.94311523  0.32742587  0.05987306
  0.67205924  0.88843513  0.34179601  0.06025261  0.77868128  0.67401105
  0.10349388  0.53170758  0.29022989  0.54501963  0.43309814  0.77815181
  0.75175041  0.00977294  0.35255209  0.29034185  0.08576855  0.52460378
  0.23889856  0.72486675  0.69800818  0.04585781  0.61279339  0.50740743
  0.68444997  0.34917989  0.84150833  0.60718584  0.92187607  0.06784791
  0.05042362  0.241505    0.10242676  0.75916296  0.69841975  0.73618746
  0.50816816  0.00163782  0.38006333  0.81526124  0.9681828   0.8107264
  0.20101748  0.21990283  0.09143824  0.11925177  0.1662218   0.12141241
  0.67629176  0.61114168  0.84703767  0.77604914  0.06363823  0.66954446
  0.9632895   0.48910257  0.81503087  0.48234224  0.83315003  0.87492347
  0.2759271   0.49929336  0.20392086  0.36211187  0.81115526  0.62664509
  0.66678047  0.73108047  0.13876791  0.24850345  0.60703003  0.81115836
  0.27531698  0.35072556  0.44352838  0.24446779  0.41860369  0.19653501
  0.90031695  0.73517466  0.41834632  0.36447218  0.32926932  0.46669638
  0.54500341  0.19840696  0.63673937  0.01909828]
[ 5.28951645  4.03795528  3.49740481  3.12685609  4.03350401  3.41349196
  3.88704538  3.82160521  4.79440451  5.77569675  3.76205945  3.12333083
  4.79578209  5.5661869   3.80041647  3.12413549  5.16370678  4.80231285
  3.21769881  4.34612799  3.66469312  4.38708591  4.05377007  5.16182375
  5.06862926  3.0196414   3.8293972   3.66498208  3.17889333  4.32441664
  3.53486967  4.97516537  4.88323164  3.09381866  4.60110235  4.27227688
  4.83737183  3.82028627  5.39115286  4.58304644  5.69360733  3.14029908
  3.10338974  3.54133463  3.21534467  5.09465408  4.88462973  5.01434708
  4.27457142  3.00327826  3.90457487  5.29517365  5.87374353  5.27873039
  3.44244289  3.48816299  3.19123745  3.25272465  3.36007333  3.25756574
  4.80995417  4.59577751  5.41154814  5.15435028  3.1313262   4.78737879
  5.85450554  4.2174263   5.29433727  4.19733858  5.3604393   5.51533794
  3.62799001  4.24788046  3.44942546  3.85534859  5.28028345  4.64597416
  4.77815723  4.99663973  3.29679227  3.55876088  4.58254528  5.28029442
  3.62643337  3.82445955  4.08377409  3.54870009  4.01243639  3.43169594
  5.61120462  5.01083088  4.01170635  3.86178446  3.76695681  4.15119839
  4.38703537  3.43617916  4.67891598  3.03856134]
```

然后我们使用这个训练集完成一个函数拟合的训练，我们这次使用的是虚构的训练集，我想，如果我们可以从外部获取训练集，那我们就可以做任何已知x与y离散点的拟合工作。完整程序：

```python
# -*- coding: utf-8 -*-
#设计一个函数拟合程序
import tensorflow as tf
import numpy as np


#首先生成一个训练集，主要就是100个点。
#生成x。这个是一个1-100的随机数列
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*x_data + 2*x_data + 3

#print x_data
# print y_data

#开始创建tensorflow结构，A、B、C分别是二次函数的三个参数
#我们设定三个参数的初始值范围，以及维度
A= tf.Variable(tf.random_uniform([1],-2,2))
B= tf.Variable(tf.random_uniform([1],-3,3))
C= tf.Variable(tf.random_uniform([1],-4,4))

#使用上面定义的参数以及x_data组成带参的二次函数，这个y是就是一个y_data的预测值
y = A*x_data*x_data + B*x_data + C

#计算y与y_data相比的误差，reduce_mean计算出平局值，square计算的是y-ydata的平方，应该
loss = tf.reduce_mean(tf.square(y - y_data))

#创建一个优化器，GradientDescentOptimizer拥有一种非常基础的优化策略，梯度下降算法，形参应该是类似于预测步长之类的参数
#我觉得如果这里的学习效率越低应该需要的训练次数就越多，但是训练的就会越精准。如果这个值太大就会出问题的
opt = tf.train.GradientDescentOptimizer(0.1)

#创建一个训练器，使用梯度下降算法减少y_data与y之间的误差
trainer = opt.minimize(loss)

#在神经网络中初始化变量
init = tf.initialize_all_variables()

#这里tensorflow结构就建立完了

#创建一个会话，应该就是一个任务的意思
sess = tf.Session();

#使用初始化器激活这个训练任务
sess.run(init)

#从这里开始训练
for stepNum in range(100000):
    sess.run(trainer)#执行一次训练
    #每10部打印出来一个训练的结果
    if stepNum % 10 == 0:
        #run这个函数的意义应该就是在神经网络中执行一些东西，执行变量就会返回变量名
        print "stepNum =", stepNum ,sess.run(A),sess.run(B),sess.run(C)


```

通过我们的训练，我们就可以发现最后拟合的结果已经非常接近了。三个参数应该是1，2，3.这里已经非常接近，我们训练的步长定为0.1，训练了100000次

```shell
stepNum = 99990 [ 1.00010967] [ 1.99988365] [ 3.00002098]
```

而且我们发现16000次左右这个拟合的值就不变化了，我觉得是`tf.train.GradientDescentOptimizer()`形参值还是太大了导致的。

## 在Linux虚拟机中进行TensorFlow serving的安装

TensorFlow Serving是一个为机器学习模型服务器的开源软件。他应该是一个CS结构的东西，我们可以将一个已经训练好的机器学习模型放到server端，然后提供服务器。在我看来，TensorFlow Serving提供比较好的版本管理。我们可以在同一时间在Server上使用不同的Model，甚至不同版本的相同Model。总之我觉得是一个Model的部署平台。

### TensorFlow Serving的安装

我们根据[TensorFlow Serving Install](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)提供的内容来进行TensorFlow Serving Install的安装。TensorFlow应该使用的是编译安装的方式。

首先我们需要安装Bazel，这个东西的作用类似于cmake，是一种构建工具。我们首先先下载这个玩意[[下载链接](https://github-cloud.s3.amazonaws.com/releases/20773773/08e8cd6a-0a52-11e7-8fb2-128a1bec9d0a.sh?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAISTNZFOVBIJMK3TQ%2F20170321%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20170321T061725Z&X-Amz-Expires=300&X-Amz-Signature=27dccdbec04e8e954b103228303597ef15762e52efcc6c19f36807fe8029a4dc&X-Amz-SignedHeaders=host&actor_id=12743988&response-content-disposition=attachment%3B%20filename%3Dbazel-0.4.5-installer-linux-x86_64.sh&response-content-type=application%2Foctet-stream)]。第一安装发现没有Java环境，好吧，醉了。我们将JDK放在/usr/local下，并且设定好了环境变量。然后我们再执行Bazel的安装脚本，完成安装：

```shell
zhendu@ubuntu:~/Desktop$ ./bazel-0.4.5-installer-linux-x86_64.sh --user
Bazel installer
---------------

Bazel is bundled with software licensed under the GPLv2 with Classpath exception.
You can find the sources next to the installer on our release page:
   https://github.com/bazelbuild/bazel/releases

# Release 0.4.5 (2017-03-16)

Baseline: 2e689c29d5fc8a747216563235e905b1b62d63b0

Cherry picks:
   + a28b54033227d930672ec7f2714de52e5e0a67eb:
     Fix Cpp action caching
   + 6d1d424b4c0da724e20e14235de8012f05c470f8:
     Fix paths of binaries in .deb packages.
   + 0785cbb672357d950e0c045770c4567df9fbdc43:
     Update to guava 21.0 and Error Prone version 2.0.18-20160224
   + 30490512eb0e48a3774cc4e4ef78680e77dd4e47:
     Update to latest javac and Error Prone
   + 867d16eab3bfabae070567ecd878c291978ff338:
     Allow ' ', '(', ')' and '$' in labels
   + 7b295d34f3a4f42c13aafc1cc8afba3cb4aa2985:
     Pass through -sourcepath to the JavaBuilder
   + 14e4755ce554cdfc685fc9cc2bfb5b699a3b48f4:
     PathFragment comparisons are now platform-aware
   + ed7795234ca7ccd2567007f2c502f853cd947e50:
     Flag to import external repositories in python import path
   + 81ae08bbc13f5f4a04f18caae339ca77ae2699c1:
     Suppress error for non-exhaustive switches
   + e8d1177eef9a9798d2b971630b8cea59471eec33:
     Correctly returns null if an environment variables is missing
   + 869d52f145c077e3499b88df752cebc60af51d66:
     Fix NPE in Android{S,N}dkRepositoryFunction.
   + d72bc57b60b26245e64f5ccafe023a5ede81cc7f:
     Select the good guava jars for JDK7 build
   + 92ecbaeaf6fa11dff161254df38d743d48be8c61:
     Windows: Assist JNI builds with a target for jni_md.h.
   + 36958806f2cd38dc51e64cd7bcc557bd143bbdb6:
     Add java_common.create_provider to allow creating a
     java_common.provider
   + 8c00f398d7be863c4f502bde3f5d282b1e18f504:
     Improve handling of unknown NDK revisions in
     android_ndk_repository.
   + b6ea0d33d3ab72922c8fb3ec1ff0e437af09584d:
     Add the appropriate cxx_builtin_include_directory entries for
     clang to the Android NDK crosstool created by
     android_ndk_repository.

Incompatible changes:

  - Depsets (former sets) are converted to strings as "depset(...)"
    instead of
    "set(...)".
  - Using --symlink_prefix is now applied to the output
    symlink (e.g. bazel-out) and the exec root symlink (e.g.
    bazel-workspace).
  - Bazel now uses the test's PATH for commands specified as
        --run_under; this can affect users who explicitly set PATH to
    a more
        restrictive value than the default, which is to forward the
    local PATH
  - It's not allowed anymore to compare objects of different types
    (i.e. a string to an integer) and objects for which comparison
    rules are not
    defined (i.e. a dict to another dict) using order operators.

New features:

  - environ parameter to the repository_rule function let
    defines a list of environment variables for which a change of
    value
    will trigger a repository refetching.

Important changes:

  - android_ndk_repository now supports Android NDK R13.
  - Android resource shrinking is now available for android_binary
    rules. To enable, set the attribute 'shrink_resources = 1'. See
    https://bazel.build/versions/master/docs/be/android.html#android_b
    inary.shrink_resources.
  - resolve_command/action's input_manifest return/parameter is now
    list
  - For increased compatibility with environments where UTS
    namespaces are not available, the Linux sandbox no longer hides
    the hostname of the local machine by default. Use
    --sandbox_fake_hostname to re-enable this feature.
  - proto_library: alias libraries produce empty files for descriptor
    sets.
  - Adds pkg_rpm rule for generating RPM packages.
  - Allow CROSSTOOL files to have linker flags specific to static
    shared libraries.
  - Make it mandatory for Java test suites in bazel codebase, to
    contain at least one test.
  - Support for Java 8 lambdas, method references, type annotations
    and repeated annotations in Android builds with
    --experimental_desugar_for_android.
  - Removed .xcodeproj automatic output from objc rules. It can still
    be generated by requesting it explicitly on the command line.
  - Flips --explicit_jre_deps flag on by default.
  - Activate the "dbg", "fastbuild", and "opt" features in the objc
    CROSSTOOL.
  - Remove support for configuring JDKs with filegroups; use
    java_runtime and java_runtime_suite instead
  - android_ndk_repository api_level attribute is now optional. If not
    specified, the highest api level in the ndk/platforms directory
    is used.

## Build informations
   - [Build log](http://ci.bazel.io/job/Bazel/JAVA_VERSION=1.8,PLATFORM_NAME=linux-x86_64/1378/)
   - [Commit](https://github.com/bazelbuild/bazel/commit/037b9b9)
Uncompressing......Extracting Bazel installation...
.

Bazel is now installed!

Make sure you have "/home/zhendu/bin" in your path. You can also activate bash
completion by adding the following line to your :
  source /home/zhendu/.bazel/bin/bazel-complete.bash

See http://bazel.build/docs/getting-started.html to start a new project!
zhendu@ubuntu:~/Desktop$ 
```

安装成功之后我们应该就会在当前用户home文件夹下看见一个bin/bazel

```shell
zhendu@ubuntu:~/Desktop$ cd ~
zhendu@ubuntu:~$ ls
bin      Documents  examples.desktop  Pictures  Templates
Desktop  Downloads  Music             Public    Videos
zhendu@ubuntu:~$ cd bin/
zhendu@ubuntu:~/bin$ ls
bazel
zhendu@ubuntu:~/bin$ 
```

然后我们把这个东西加入环境变量。

此外TensorFlow Serving应该还依赖gRPC，他是一个远程调用协议。我猜测是提供TensorFlow Serving服务器与客户端之间的通信。我们需要安装实现了这个协议的Python包。我打算在整个系统的层面安装这个包，使用pip命令。

```shell
sudo pip install grpcio
```

但是安装完之后最大的问题就是竟然一点python运行import的时候找不到这个模块。估计是个隐患。

之后我们就可以安装所有零碎的Linux依赖包了。

现在所有的前期依赖就完成了。

然后我们从github上面克隆TensorFlow Serving的代码。并进行编译安装。在安装之前，我们前往了原码的TensorFlow文件夹，运行里面的config，并且进行TensorFlow的配置，配置结束之后就会进入编译阶段。我们不安装任何依赖，最后的输出是这样的。

```shell
INFO: All external dependencies fetched successfully.
```

当这一步之后我们就要进行TensorFlow Serving的安装。他依赖于我们上面的TensorFlow。编译完的所有二进制程序会存放在bazel-bin目录下。但是无奈啊，总是报这个错。

```shell
ERROR: /home/zhendu/.cache/bazel/_bazel_zhendu/bd849f9b90e223f76b575a2ac1899a66/external/org_tensorflow/tensorflow/core/kernels/BUILD:2136:1: C++ compilation of rule '@org_tensorflow//tensorflow/core/kernels:matmul_op' failed: gcc failed: error executing command /usr/bin/gcc -U_FORTIFY_SOURCE -fstack-protector -Wall -B/usr/bin -B/usr/bin -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer '-std=c++0x' -MD -MF ... (remaining 93 argument(s) skipped): com.google.devtools.build.lib.shell.BadExitStatusException: Process exited with status 1.
virtual memory exhausted: Cannot allocate memory
INFO: Elapsed time: 699.430s, Critical Path: 608.91s
zhendu@ubuntu:~/serving$ 

```

我们扩大了内存容量，继续执行编译。扩了3次内存，竟然终于可以过了，4核12G。

然后我们执行bazel的test命令，通过了，至此安装成功。

当安装完成之后，我们就可以使用MNIST数据集进行一个测试了，MNIST是个手写体图片集，TensorFlow做的工作就是把这个数据集的每一张图片和一个手写的数字对应起来。X是一个图片对应的矩阵，Y是这个图片对应的数字。

### 测试TensorFlow Serving的服务器端

首先我们使用example文件夹里面的程序来生成Model，我们按照教程总是出错。结果我干脆使用pip卸载了TensorFlow。然后安装了0.12版本的Tensorflow，然后编译`//tensorflow_serving/example:mnist_saved_model`编译完之后就终于可以正常运行了。但是运行之后又出错，而且是拒绝访问。

```shell
IOError: [Errno socket error] [Errno 111] Connection refused
```

我把VPN关掉再试试。还是拒绝访问，我从谷歌直接搜索要访问的域名，结果发现是这个网站挂了。

我们使用其他网站。下载成功并且可以使用了。并且我们在这里看到了训练的成果。

```shel
zhendu@ubuntu:~/serving$ cd /tmp/mnist_model/
zhendu@ubuntu:/tmp/mnist_model$ ls
1
zhendu@ubuntu:/tmp/mnist_model$ cd 1
zhendu@ubuntu:/tmp/mnist_model/1$ ls
saved_model.pb  variables
```

下面我们编译了服务器。在开启服务器时，设定模型的基础目录为/tmp/mnist_model，端口为9000。打开之后这个终端就会被暂时占用。

```shell
zhendu@ubuntu:~/serving$ bazel build //tensorflow_serving/model_servers:tensorflow_model_server
.................................
2017-03-21 08:35:03.123085: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:239] Loading SavedModel: success. Took 57580 microseconds.
2017-03-21 08:35:03.123167: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: mnist version: 1}
2017-03-21 08:35:03.135204: I tensorflow_serving/model_servers/main.cc:272] Running ModelServer at 0.0.0.0:9000 ...
```

然后我们编译执行客户端。并让这个客户端发送1000个样本去让服务器中的model判断，并且客户端会判断服务器说的对不对。给出一个错误率。我们看到错误率是10.4%。

```shell
zhendu@ubuntu:~/serving$ bazel build //tensorflow_serving/example:mnist_client
WARNING: /home/zhendu/.cache/bazel/_bazel_zhendu/bd849f9b90e223f76b575a2ac1899a66/external/org_tensorflow/tensorflow/workspace.bzl:72:5: tf_repo_name was specified to tf_workspace but is no longer used and will be removed in the future.
WARNING: /home/zhendu/.cache/bazel/_bazel_zhendu/bd849f9b90e223f76b575a2ac1899a66/external/org_tensorflow/tensorflow/contrib/learn/BUILD:15:1: in py_library rule @org_tensorflow//tensorflow/contrib/learn:learn: target '@org_tensorflow//tensorflow/contrib/learn:learn' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:exporter': Use SavedModel Builder instead.
WARNING: /home/zhendu/.cache/bazel/_bazel_zhendu/bd849f9b90e223f76b575a2ac1899a66/external/org_tensorflow/tensorflow/contrib/learn/BUILD:15:1: in py_library rule @org_tensorflow//tensorflow/contrib/learn:learn: target '@org_tensorflow//tensorflow/contrib/learn:learn' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:gc': Use SavedModel instead.
INFO: Found 1 target...
Target //tensorflow_serving/example:mnist_client up-to-date:
  bazel-bin/tensorflow_serving/example/mnist_client
INFO: Elapsed time: 0.337s, Critical Path: 0.01s
zhendu@ubuntu:~/serving$ bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
in main
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
..........................................................................................
Inference error rate: 10.4%

```

这样子整个流程就好了。我们现在尝试自己去搭建一个客户端

## Client的编写

为了可以自己写Client，我们需要了解两个东西，一个是模型是怎么建的，还有一个是客户端是怎么写的，我想达到的效果是根据已有的mnist Model，我们写一个客户端，这个客户端可以接收一个图片，完成图片的建模，并且交给服务器端去识别。服务器返回识别之后的值。

### IDX文件解析

为了可以自己建模我们需要知道IDX文件到底里面是什么东西，我觉得应该是一个全是010101010的二进制码，对应的位数有不同的含义。主要包括的应该是图片每一个像素点RGB值的二进制表现。

我没有找到如何将img转化为IDX。但我找到了一段使用PLT和numpy的Python程序来将IDX转化为图片的Python代码。测试完之后还真是有用。这有助于我们了解IDX文件里面到底是什么，谁叫官网挂了呢。

```python
# encoding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
train_images_idx3_ubyte_file = 'train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print train_labels[i]
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print 'done'

if __name__ == '__main__':
    run()
```

然后我又找到了[Python使用struct处理二进制](http://www.cnblogs.com/gala/archive/2011/09/22/2184801.html)这篇文章。然后我打算编写一个代码，这个代码的作用就是编写只有一个数据的数据集然后使用上面这个程序测试一下，看看能不能把相同的图片读出来，以此来验证我建模的正确性。

```python
# encoding: utf-8

import numpy as np
import struct
import ctypes
import matplotlib.pyplot as plt
from PIL import Image
import binascii


def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()

if __name__=="__main__":
    filename = "./figure.png"
    data = ImageToMatrix(filename)
    #创建IDX文件
    offset = 0
    fmt_header = '>iiii'
    img_size = 28 * 28
    fmt_img = '>' + str(img_size) + 'B'
    # print struct.calcsize(fmt_header)
    buf = ctypes.create_string_buffer(struct.calcsize(fmt_header)+struct.calcsize(fmt_img))

    #写入描述训练集的第一行
    struct.pack_into(fmt_header,buf , offset ,2051 , 1 , 28 , 28)
    #偏移量按字节算
    offset = offset+struct.calcsize(fmt_header)
    print offset
    #然后写入一张图片
    data = data.reshape(1,img_size)
    print data
    #将线性矩阵转化为数组
    arr = data.getA1()

    struct.pack_into(fmt_img, buf, offset , *arr)
    # print binascii.hexlify(buf)
    with open("./my-idx3-ubyte","wb") as f:
        f.write(buf)
```

我的输入是一张图片![](https://ww1.sinaimg.cn/large/006tKfTcgy1fdvc6c2gptj300s00sa9t.jpg)

输出是一个IDX文件。我们使用IDX->img这个程序进行了测试，发现输出是一样的。现在至少说明我的建模是没有问题的。下面我们分析客户端的代码，我需要知道服务器需要什么东西，这样子才能知道自己的Client怎么写。

### TensorFlow的客户端代码

经过初步的阅读，我觉得客户端代码里面最重要的是do_inference函数，他负责数据集的发送和结果的接收。

```python
def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  for _ in range(num_tests):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    image, label = test_data_set.next_batch(1)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(label[0], result_counter))
  return result_counter.get_error_rate()
```

因为对TensorFlow Serving的api不了解，所以对我来说这就是迷之代码。不过可以看出这个函数主要负责不断发送请求并返回错误率，我觉得错误率的计算函数里面应该可以找到服务器返回的对于手写体数字的预测值。

最后我们定位到了这段代码：

```python
exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
```

这段代码中的我觉得那个prediction就是从服务器返回的手写体预测值，我们加入print语句，看看会出现什么。我们打印了返回值的数值和类型。

```shell
zhendu@ubuntu:~/serving$ bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1 --server=localhost:9000
in main
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
.7
<type 'numpy.int64'>
```

我们现在知道应该怎么改这个程序了。我们不要num_test参数，并且默认为1，并且将自己建的IDX传到服务器端。

然后我们需要知道怎么替换掉他要传的IDX。我们在Client文件中没有找到输入文件。但是我们找到了`test_data_set = mnist_input_data.read_data_sets(work_dir).test`。可见Client的代码依赖于minis_input_data这个模块，我们已知work_dir作为传入参数值应该是”/tmp“。然后我们检查mnist_input_data的read_data_sets这个函数。

```python
def read_data_sets(train_dir, fake_data=False, one_hot=False):
  """Return training, validation and testing data sets."""

  class DataSets(object):
    pass

  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
    return data_sets

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets
```

train_dir就是之前传进来的路径，也就是/tmp。而maybe_download就是获取数据集。我们可以看到maybe_download就是获取数据集的过程。所以我们的程序可以做一些修改，我们首先接收一个图片的路径，然后我们将这个图片的路径上的图片建模。并且覆盖测试集IDX文件。这就需要我们在main函数的前面添加代码。接收一个路径并且建模。

### 重写TensorFlow的客户端

是时候修改代码了我们首先在前面加一段获取图片路径的代码。使用scanner。 我们将修改的代码放在了github中。我们的运行过程需要依赖PIL。

PIL的安装:

[ubuntu apt-get (You must put some ‘source’ URIs in your sources.list)](https://www.cplusplus.me/2375.html)

[PIP install PIL python2.7 ubuntu 14.04.1](http://stackoverflow.com/questions/28044332/pip-install-pil-python2-7-ubuntu-14-04-1)

我们复用之前写的img转IDX的代码，除了自己可以做出来的Image的IDX文件之外，我们还需要在mnist_data_input.py文件中把label IDX的检查去掉，因为实际上这个文件是给Client和import_model两个文件用的，而这个检查是为了导出model的时候用的。在向服务器发送测试集的时候我们不需要进行这个检查。所以我们把这个注释掉。

```python
#assert images.shape[0] == labels.shape[0], (
      #   'images.shape: %s labels.shape: %s' % (images.shape,
      #                                          labels.shape))
```

我们整个程序流程是这样的，客户端可以选择一个28*28的手写数字图片。客户端将图片转化为IDX文件，并将IDX文件上传到服务器端进行解析，服务器返回一个数字作为预测的结果。

输入与输出：

```shell
zhendu@ubuntu:~/serving$ bazel-bin/tensorflow_serving/example/mnist_client --server=localhost:9000
请输入要判别的文件:/home/zhendu/Desktop/figure.png
.预测的数字 5
zhendu@ubuntu:~/serving$ 
```

我会将Client项目文件夹上传到Github上。

## TensorFlow Serving进阶

如果我们需要使用Kubernetes做负载均衡，根据官方文档，我们需要了解官方文档中的进阶教程。我们之前使用的是一个叫做TensorFlow的基础服务器。这里我们可以知道怎么写一个TensorFlow的标准服务器。

首先我们可以使用`--model_version=`参数导出两个不一样版本的Model。这样子我们就可以在我们之前放Model的文件夹/tmp/model下看到两个文件夹，一个是命名为1，一个是2。

### ServerCore

下面分析一下TensorFlow Serving的构建源码。首先是`ServerCore::Create()`这个函数。他负责初始化一个ServerCore，他是TensorFlow的核心。我们通过设定这个函数的第一个形参，ServerCore::Options，可以设定我们的Model加载办法。这个ServerCore::Options作为一个结构体包含一系列内容，比如，既可以设定开始就加载一个静态的Model列表，也可以先加载一个在运行过程中会不断变化和升级的动态Model列表，等等。

ServerCore在一开始做了这么几件事情：

- 实例化FileSystemStoragePathSource，这个东西监管model导出的目录。这个目录在model_config_list这个配置文件中声明。
- 使用PlatformConfigMap实例化SourceAdapter，并且让SourceAdapter和FileSystemStoragePathSource进行连接。PlatformConfigMap是一个键值对，注册了Model的生成所使用的平台，每个平台对应的值就是使用这个额平台所对应的配置（我的理解是使用不同平台做出来的Model是不一样的，所以对于TensorFlow Serving来说，不同平台的Model的使用方法是不一样的，就好像不同文件的打开方式是不一样的一个道理，而Model的使用方式就存在这个Map中，不同的平台查这个Map可以为TensorFlow Server使用不同的配置）。SourceAdapter和FileSystemStoragePathSource进行连接之后，在任何时候有新的Model加入导出目录的时候SavedModelBundleSourceAdapter就可以把这个Model给Loader\<SavedModelBundle>处理。Loader是SavedModelBundleSourceAdapter创建的一个对象。
- 实例化一种管理器AspiredVersionsManager。他会管理所有的Loader实例。ServerCore可以通过AspiredVersionsManager的调用导出所有的Manager接口。当诞生了一个模型的新版本，他会加载这个额新版本，并且依照默认的策略卸载旧版本。

### Batching

有关Batching（批处理）的内容，文档给的很不详细。我的猜测就是Batching为TensorFlow搭建在并行计算平台上提供方便。他提供接口，让服务器先接受一大捆数据，然后并行处理，然后返回一大捆数据的这么一种方案。

### 使用manager提供服务

我们知道，TensorFlow Serving提供Manager这种东西，他是一个Model的管理组件。他提供以下一些特性：

Servable：这是一个黑盒的对象，主要是处理客户端请求的。

Servable version：TensorFlow Serving可以服务器不同版本的Servable对象。

Servable stream：这是一系列不同版本的Servable对象的序列，版本号升序排列。

Model：这是机器学习的学习成果，相当于一个或者多个Servable对象。



看完之后，一是看得不明不白，二是好像对后面在Kubernetes中没啥软用。



## 将安装了TensorFlow Serving的容器部署在Kubernetes中

### 安装Docker与Docker中的TensorFlow Serving

这个老生常谈，直接看[Docker文档](https://docs.docker.com/engine/installation/linux/ubuntu/#uninstall-old-versions)。过了1年发现Docker的安装方法又变了。

首先我们需要添加Docker的apt仓库。然后更新apt-get的源，然后就可以开始安装了，现在的Docker分为Docker ce与Docker ee，应该前者是社区版，我们下载社区版。

在我们clone的项目中已经自带了Dockerfile。我们使用这个已有的DockerFile创建了镜像。

```shell
zhendu@ubuntu:~/serving$ sudo docker build --pull -t $USER/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
[sudo] password for zhendu: 
Sending build context to Docker daemon 525.2 MB
Step 1/13 : FROM ubuntu:14.04
```

然后我们就进行和在虚拟机中安装TensorFlow Serving一样的步骤，将TensorFlow Serving安装在Docker中。我预感又会出现爆内存的问题，我打算使用4核12G的内存分配。

安装完TensorFlow Serving之后，我们开启了Server：

```shell
root@41f4e247a169:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

教程中的代码是这样的，但是没有任何的Model导出路径和端口，端口应该是有缺省的，但是Model的存放路径我觉得还是需要设定一下。我觉得还是应该设定一下。这个留在稍后再看。

现在回到导入Model的事情，在给TensorFlow Server导Model的时候出错了`Can't import inception_model`，我们在Issue中找到了事情的原因和解决办法。[Can't import inception_model](https://github.com/tensorflow/serving/issues/354)，应该就是类似于Mnist数据集的IDX文件不存在，我们需要从TensorFlow Model的github上直接下载。然后使用软连接或者拷贝的方式把数据集放到对应的文件夹下。

在Docker Container中安装完环境之后，我们需要把这个container打包成一个新的镜像，方便之后的部署。首先执行Commit命令，这个命令等了很久还没有反应。等了一段时间之后，完成了，我们查看一下现有的镜像：

```shell
zhendu@ubuntu:~$ sudo docker images
REPOSITORY                        TAG                 IMAGE ID            CREATED             SIZE
zhendu/inception_serving          latest              2cb2087eea36        16 seconds ago      6.73 GB
<none>                            <none>              4c6756edff2c        2 minutes ago       6.73 GB
zhendu/tensorflow-serving-devel   latest              a802fb23be0b        13 hours ago        1.03 GB
ubuntu                            14.04               7c09e61e9035        3 weeks ago         188 MB
```

接下来我们测试一下这个镜像。我们打开这个镜像之后让其中的TensorFlow Serving进行一次图像的分类。我们打开这个镜像并开启TensorFlow Serving的服务器端：

```shell
zhendu@ubuntu:~$ sudo docker run -it $USER/inception_serving
root@0c7e82bd2a23:/# cd serving/
root@0c7e82bd2a23:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=inception-export &> inception_log &
[1] 14
root@0c7e82bd2a23:/serving# 
```

我们创建客户端。整个软件的流程应该是客户端发送一只猫的图片给服务器端，服务器返回一个Json来描述这只猫的信息。但是好像现在还没有图片，我去找只猫的图片试试。

![](https://ww1.sinaimg.cn/large/006tKfTcgy1fdwsoiq68qj303d03d3ya.jpg)

我们使用docker cp命令把一个图片拷贝到容器中：

```shell
zhendu@ubuntu:~$ sudo docker cp /home/zhendu/Desktop/bigcat.jpeg 0c7e82bd2a23:/
zhendu@ubuntu:~$ 
```

在容器中我们看到了这张图片：

```shell
root@0c7e82bd2a23:/serving# cd /
root@0c7e82bd2a23:/# ls
bazel        bin   dev  home  lib64  mnt     opt   root  sbin     srv  tmp  var
bigcat.jpeg  boot  etc  lib   media  models  proc  run   serving  sys  usr
root@0c7e82bd2a23:/# 
```

然后我们将这只猫传给服务器进行识别：

```shell
root@0c7e82bd2a23:/serving# bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=/bigcat.jpeg
D0323 07:23:15.457716614     814 ev_posix.c:101]             Using polling engine: poll
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "Persian cat"
    string_val: "Japanese spaniel"
    string_val: "Angora, Angora rabbit"
    string_val: "titi, titi monkey"
    string_val: "marmoset"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 8.54975223541
    float_val: 2.50879764557
    float_val: 2.4423801899
    float_val: 2.21044707298
    float_val: 1.93949532509
  }
}

E0323 07:23:18.276729296     814 chttp2_transport.c:1810]    close_transport: {"created":"@1490253798.276704125","description":"FD shutdown","file":"src/core/lib/iomgr/ev_poll_posix.c","file_line":427}
root@0c7e82bd2a23:/serving# 
```

通过`string_val: "Persian cat"`我们可以看到他真的是波斯猫。看来这个容器的运行已经没有任何问题了。

### 安装Kubernetes

我们根据教程[Ubuntu14.04单机版kubernetes安装指导原理及实践应用](http://www.linuxdown.net/install/soft/2016/0114/4362.html)搭建单机版Kubernetes环境。

首先我们在机子上安装Golang，然后下载Kubernetes的原码进行编译，编译的过程主要就是执行在build文件夹下的release.sh脚本。在一开始的时候就报错。`cannot stat 'build/build-image/Dockerfile': No such file or directory`其实这个DockerFile这个文件是存在的，只是他找的目录错了，build-image是一个同级目录，所以我们需要建立一个软连接，以便安装脚本的继续运行。

但是编译的最后老是出错，我们首先尝试加大内存看看行不行。加大内存之后编译的时间明显变长。这个世界没有+4G内存解决不了的编译器报错，如果有，那就再加4G。

过了好久，现在的状况是命令行运行在`+++ [0323 04:49:05] Running rsync`CPU占用也几乎没有了，我猜测这应该就是安装好了（其实并没有安装好，命令行又开始动了）。

最后还是报错了，虚拟机的空间不够了。所以我的处理办法是，新开一个空间足够大的虚拟机（[Docker镜像的导出和导入](http://blog.csdn.net/a906998248/article/details/46236687)），然后我们把Docker镜像拷贝过来。打开，这样可以保留今天的成果，也不用承担重新调整磁盘分区的风险。

现在我打算安装etcd，然后解压下好的文件。

```shell
wget https://github.com/coreos/etcd/releases/download/v0.4.6/etcd-v0.4.6-linux-amd64.tar.gz
```



## 在TensorFlow Serving上使用im2txt模型

这个场景也是图片，之前Mnist输入也是图片，我觉得建模方式可能会简单一点：）。因为这里im2txt没有自己写客户端，所以通过修改代码的方式已经没有可能性了，我觉得如果要写客户端需要知道3点：1、将Model找到并放在TensorFlow规定的文件夹中；2、找到Model的输入和输出是什么。3、在客户端中使用TensorFlow Serving的接口

























