# 初始化计数器
retry_count=0

# 无限循环直到 train.py 成功执行
while true; do
    # 增加计数器
    ((retry_count++))

    # 运行 train.py 并捕获退出码
    python train.py
    status=$?

    # 打印执行次数
    echo "Train script execution attempt $retry_count:"

    # 检查 train.py 是否成功执行（退出码为 0）
    if [ $status -eq 0 ]; then
        echo "train.py executed successfully. Proceeding to run test.py..."

        # 运行 test.py
        python test.py
        if [ $? -eq 0 ]; then
            echo "test.py executed successfully."
        else
            echo "test.py failed."
        fi
        break  # 成功后退出循环
    else
        echo "train.py failed with status $status. Retrying in 5 seconds..."
        sleep 5
    fi
done
