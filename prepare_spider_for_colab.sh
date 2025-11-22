#!/bin/bash

# Script to prepare Spider dataset for Colab upload
# Usage: ./prepare_spider_for_colab.sh

echo "🚀 Preparing Spider dataset for Colab upload..."
echo ""

# Navigate to data directory
cd "$(dirname "$0")/backend/vector_service/data" || exit 1

# Check if spider folder exists
if [ ! -d "spider" ]; then
    echo "❌ Error: spider folder not found at backend/vector_service/data/spider"
    echo "Please ensure the dataset is downloaded first."
    exit 1
fi

# Check for required files
echo "📋 Checking required files..."
required_files=("tables.json" "train_spider.json" "train_others.json" "dev.json")
all_files_exist=true

for file in "${required_files[@]}"; do
    if [ -f "spider/$file" ]; then
        size=$(du -h "spider/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file - MISSING"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "❌ Some required files are missing. Cannot proceed."
    exit 1
fi

echo ""
echo "📦 Creating spider.zip..."

# Remove old zip if exists
if [ -f "spider.zip" ]; then
    echo "  Removing old spider.zip..."
    rm spider.zip
fi

# Create ZIP (exclude database files if they exist)
zip -r spider.zip spider/ \
    -x "*.sqlite" \
    -x "*/__pycache__/*" \
    -x "*/.DS_Store" \
    -q

if [ $? -eq 0 ]; then
    zip_size=$(du -h spider.zip | cut -f1)
    echo "  ✓ Created spider.zip ($zip_size)"
    echo ""
    echo "✅ Success! spider.zip is ready for upload to Colab."
    echo ""
    echo "📍 Location: $(pwd)/spider.zip"
    echo ""
    echo "📝 Next steps:"
    echo "  1. Open Google Colab and create a new notebook"
    echo "  2. Enable GPU: Runtime → Change runtime type → T4 GPU"
    echo "  3. Run the upload cell from COLAB_FINETUNE.md"
    echo "  4. Select spider.zip when prompted"
    echo ""
else
    echo "❌ Error creating ZIP file"
    exit 1
fi
