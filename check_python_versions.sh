#!/bin/bash
# Script to check available Python versions

echo "🔹 Checking available Python versions on the system..."

# Check common Python commands
python_versions=("python" "python3" "python3.6" "python3.7" "python3.8" "python3.9" "python3.10" "python3.11" "python3.12")

echo "Available Python versions:"
echo "=========================="

for version in "${python_versions[@]}"; do
    if command -v $version &> /dev/null; then
        echo -n "✅ $version: "
        $version --version 2>/dev/null || echo "version info not available"
    else
        echo "❌ $version: not found"
    fi
done

echo ""
echo "🔹 Checking which Python is the default:"
echo "=========================="
echo "python command: $(which python)"
echo "python3 command: $(which python3)"

echo ""
echo "🔹 Checking module availability:"
echo "=========================="
if command -v module &> /dev/null; then
    echo "✅ module command available"
    echo "Available Python modules:"
    module avail python 2>/dev/null | grep -i python || echo "No Python modules found"
else
    echo "❌ module command not available"
fi

echo ""
echo "🔹 Checking conda availability:"
echo "=========================="
if command -v conda &> /dev/null; then
    echo "✅ conda available"
    conda info --envs
else
    echo "❌ conda not available"
fi 