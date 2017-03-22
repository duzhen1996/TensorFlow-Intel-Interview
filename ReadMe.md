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
WARNING: /home/zhendu/.cache/bazel/_bazel_zhendu/bd849f9b90e223f76b575a2ac1899a66/external/org_tensorflow/tensorflow/workspace.bzl:72:5: tf_repo_name was specified to tf_workspace but is no longer used and will be removed in the future.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:561:1: in cc_library rule //tensorflow_serving/servables/tensorflow:regressor: target '//tensorflow_serving/servables/tensorflow:regressor' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:561:1: in cc_library rule //tensorflow_serving/servables/tensorflow:regressor: target '//tensorflow_serving/servables/tensorflow:regressor' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:signature': Use SavedModel instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:523:1: in cc_library rule //tensorflow_serving/servables/tensorflow:classification_service: target '//tensorflow_serving/servables/tensorflow:classification_service' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:419:1: in cc_library rule //tensorflow_serving/servables/tensorflow:get_model_metadata_impl: target '//tensorflow_serving/servables/tensorflow:get_model_metadata_impl' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:214:1: in cc_library rule //tensorflow_serving/servables/tensorflow:session_bundle_source_adapter: target '//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:123:1: in cc_library rule //tensorflow_serving/servables/tensorflow:session_bundle_factory: target '//tensorflow_serving/servables/tensorflow:session_bundle_factory' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:498:1: in cc_library rule //tensorflow_serving/servables/tensorflow:classifier: target '//tensorflow_serving/servables/tensorflow:classifier' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:498:1: in cc_library rule //tensorflow_serving/servables/tensorflow:classifier: target '//tensorflow_serving/servables/tensorflow:classifier' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:signature': Use SavedModel instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:523:1: in cc_library rule //tensorflow_serving/servables/tensorflow:classification_service: target '//tensorflow_serving/servables/tensorflow:classification_service' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:signature': Use SavedModel instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:399:1: in cc_library rule //tensorflow_serving/servables/tensorflow:predict_impl: target '//tensorflow_serving/servables/tensorflow:predict_impl' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:399:1: in cc_library rule //tensorflow_serving/servables/tensorflow:predict_impl: target '//tensorflow_serving/servables/tensorflow:predict_impl' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:signature': Use SavedModel instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:542:1: in cc_library rule //tensorflow_serving/servables/tensorflow:regression_service: target '//tensorflow_serving/servables/tensorflow:regression_service' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle': Use SavedModel Loader instead.
WARNING: /home/zhendu/serving/tensorflow_serving/servables/tensorflow/BUILD:542:1: in cc_library rule //tensorflow_serving/servables/tensorflow:regression_service: target '//tensorflow_serving/servables/tensorflow:regression_service' depends on deprecated target '@org_tensorflow//tensorflow/contrib/session_bundle:signature': Use SavedModel instead.
INFO: Found 1 target...
Target //tensorflow_serving/model_servers:tensorflow_model_server up-to-date:
  bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
INFO: Elapsed time: 4.384s, Critical Path: 3.68s
zhendu@ubuntu:~/serving$ bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
2017-03-21 08:35:02.963063: I tensorflow_serving/model_servers/main.cc:152] Building single TensorFlow model file config:  model_name: mnist model_base_path: /tmp/mnist_model/ model_version_policy: 0
2017-03-21 08:35:02.963360: I tensorflow_serving/model_servers/server_core.cc:337] Adding/updating models.
2017-03-21 08:35:02.963399: I tensorflow_serving/model_servers/server_core.cc:383]  (Re-)adding model: mnist
2017-03-21 08:35:03.065254: I tensorflow_serving/core/basic_manager.cc:698] Successfully reserved resources to load servable {name: mnist version: 1}
2017-03-21 08:35:03.065353: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: mnist version: 1}
2017-03-21 08:35:03.065381: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: mnist version: 1}
2017-03-21 08:35:03.065478: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /tmp/mnist_model/1
2017-03-21 08:35:03.065503: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:195] Loading SavedModel from: /tmp/mnist_model/1
2017-03-21 08:35:03.068896: W external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-03-21 08:35:03.068934: W external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-03-21 08:35:03.068944: W external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-03-21 08:35:03.068949: W external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-03-21 08:35:03.068954: W external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-03-21 08:35:03.112036: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:114] Restoring SavedModel bundle.
2017-03-21 08:35:03.119241: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:149] Running LegacyInitOp on SavedModel bundle.
2017-03-21 08:35:03.123085: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:239] Loading SavedModel: success. Took 57580 microseconds.
2017-03-21 08:35:03.123167: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: mnist version: 1}
2017-03-21 08:35:03.135204: I tensorflow_serving/model_servers/main.cc:272] Running ModelServer at 0.0.0.0:9000 ...

```

然后我们编译执行客户端。并让这个客户端发送1000个样本去让服务器中的model判断，并且客户端会判断服务器说的对不对。给出一个正确率。我们看到正确率是10.4%。

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


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张' % (magic_number, num_images)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


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


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
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


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
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

输出是一个IDX文件。我们使用IDX->img这个程序进行了测试，现在至少说明我的建模是没有问题的。下面我们分析客户端的代码，我需要知道服务器需要什么东西，这样子才能知道自己的Client怎么写。

### TensorFlow的客户端代码





















