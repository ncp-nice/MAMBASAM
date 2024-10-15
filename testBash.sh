command="nohup python main.py --action "train&test" --arch UNet --epoch 21 --batch_size 1"

# 初始化计数器
retry_count=0

# 无限循环直到命令成功执行
while true; do
    # 增加计数器
    ((retry_count++))

    # 运行命令并捕获退出码
    $command
    status=$?

    # 打印执行次数
    echo "脚本执行次数Attempt $retry_count:"

    # 检查命令是否成功执行（退出码为 0）
    if [ $status -eq 0 ]; then
        echo "Command executed successfully."
        break
    else
        echo "Command failed with status $status. Retrying in 5 seconds..."
        sleep 5
    fi
done

