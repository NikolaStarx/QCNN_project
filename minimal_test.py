#!/usr/bin/env python3
# minimal_test.py - 最小测试

print("🔍 Testing QCNN components...")

try:
    # 测试导入
    from models.qcnn_fixed import QCNN
    from encoders.amplitude import build_amplitude_encoder_circuit
    import torch
    print("✅ Imports successful")
    
    # 测试模型创建
    print("🧠 Creating QCNN model...")
    model = QCNN(
        num_qubits=4,  # 用更小的尺寸测试
        encoder_fn=build_amplitude_encoder_circuit,
        num_classes=2
    )
    print("✅ Model created")
    
    # 测试前向传播
    print("🔄 Testing forward pass...")
    dummy_input = torch.randn(1, 16)  # 2^4 = 16 维
    dummy_input = dummy_input / torch.norm(dummy_input)  # 归一化
    
    output = model(dummy_input)
    print(f"✅ Forward pass successful, output shape: {output.shape}")
    print(f"📊 Output: {output}")
    
    print("\n🎉 All tests passed! QCNN is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
