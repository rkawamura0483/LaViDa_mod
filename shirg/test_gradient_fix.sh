#!/bin/bash

echo "🧪 Testing SHIRG LoRA Gradient Fix"
echo "=================================="

# Test gradient fix
echo ""
echo "Running gradient fix test..."
python shirg/test_gradient_fix.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Gradient fix test PASSED!"
    echo ""
    echo "You can now run the full training with:"
    echo "  bash shirg/test_8gpu.sh"
else
    echo ""
    echo "❌ Gradient fix test FAILED!"
    echo "Please check the error messages above."
    exit 1
fi