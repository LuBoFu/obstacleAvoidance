#!/usr/bin/fish

# 张波使用该脚本将更新的二进制程序以及相关的数据文件上传到sw服务器上。
# 在张波的开发环境中，开发机和sw位于同一个局域网，可以直接通过ssh/scp命令和sw机器交互。
# 而其他成员只能使用frp中转来访问sw机器，所使用的ssh/scp命令的格式会不同。本脚本能够处理这种差异。
# 通过设置以下变量来表明开发者如何能够抵达sw机器
set useFrp    false

# 另外，本脚本是张波撰写的，脚本中的用户名为“zb”，每个开发者应该将这个名字替换为自己在sw机器上的
# 用户名。还有，本脚本假设frp所依赖的公网服务器在/etc/hosts中已经被命名为“tx”。

if test $useFrp = "false"
    set sshCommand ssh
    set scpCommand scp
    set serverName "sw"
else
    # 以下的端口号5000是使用frp访问sw机器时的端口号  
    set sshCommand ssh -oPort=5000
    set scpCommand scp -P 5000
    set serverName "tx"
end   

# 以下是和具体任务相关的内容
set targetDirectory /sw/obstacleAvoidance/clusterByDBSCAN

# create directories on sw machine.
$sshCommand zb@$serverName mkdir -p  $targetDirectory/bin
$sshCommand zb@$serverName mkdir -p  $targetDirectory/config

# copy binaries and associated files to sw machine.
$scpCommand  bin/*      zb@$serverName:$targetDirectory/bin
$scpCommand  config/*   zb@$serverName:$targetDirectory/config
$scpCommand  run        zb@$serverName:$targetDirectory
