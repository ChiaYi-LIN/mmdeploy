export LD_LIBRARY_PATH=/service/software/cuda/cuda-10.2/lib64:/tmp2/linchiayi/cuda/lib64:/tmp2/linchiayi/TensorRT-8.2.3.0-cuda10.2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/service/software/cuda/cuda-11.3/lib64:/service/software/cudnn/cudnn-11.3-linux-x64-v8.2.1.32/cuda/lib64:/tmp2/linchiayi/mmdeploy/TensorRT-8.2.3.0-cuda11.4/lib:$LD_LIBRARY_PATH

deploy_cfg="mmdeploy/configs/mmseg/segmentation_tensorrt_static-512x1024.py"
model_cfg="mmsegmentation/configs/entextnet/entextnet-stdc1_4x12_512x1024_scale0.5_160k_cityscapes.py"
model="mmdeploy/mmdeploy_model/entextnet"

python3 mmdeploy/tools/test.py \
    ${deploy_cfg} \
    ${model_cfg} \
    --model "${model}/end2end.engine" \
    --metrics mIoU \
    --device cuda \
    --log2file "${model}/logs.log" \
    --json-file "${model}/results.json" \
    --speed-test \