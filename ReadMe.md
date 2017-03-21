# TensorFlow-Intel-Interview

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

















