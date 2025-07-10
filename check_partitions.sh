#!/bin/bash
# Script to check available SLURM partitions

echo "🔹 Checking available SLURM partitions..."

# Check if sinfo is available
if command -v sinfo &> /dev/null; then
    echo "Available partitions:"
    echo "===================="
    sinfo -o "%P %D %T %C %m %G" | head -20
else
    echo "❌ sinfo command not available"
fi

echo ""
echo "🔹 Checking partition details..."
if command -v scontrol &> /dev/null; then
    echo "Partition details:"
    echo "=================="
    scontrol show partition | grep -E "PartitionName|State|Nodes|CoresPerNode|MemPerNode|MaxTime" | head -20
else
    echo "❌ scontrol command not available"
fi

echo ""
echo "🔹 Checking current partition usage..."
if command -v squeue &> /dev/null; then
    echo "Current jobs by partition:"
    echo "========================="
    squeue -o "%.10P %.8j %.8u %.2t %.10M %.6D %R" | head -10
else
    echo "❌ squeue command not available"
fi 