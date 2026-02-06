#!/bin/bash
# Live training monitor - updates every 5 seconds

echo "🎯 Starting live training monitor..."
echo "Press Ctrl+C to stop watching"
echo ""
sleep 2

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          🌙 OVERNIGHT TRAINING LIVE MONITOR               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check if training is running
    if ps aux | grep -q "[c]ontinuous_training.py"; then
        echo "✅ Status: TRAINING IS RUNNING"
        
        # Get process info
        pid=$(ps aux | grep "[c]ontinuous_training.py" | awk '{print $2}')
        cpu=$(ps aux | grep "[c]ontinuous_training.py" | awk '{print $3}')
        mem=$(ps aux | grep "[c]ontinuous_training.py" | awk '{print $4}')
        runtime=$(ps -p $pid -o etime= | tr -d ' ')
        
        echo "📊 Process: PID $pid | CPU: ${cpu}% | Memory: ${mem}% | Runtime: $runtime"
    else
        echo "⏹️  Status: TRAINING STOPPED OR FINISHED"
    fi
    
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    
    # Count labels
    manual_count=$(find detection-model/real_labels -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
    auto_count=$(find detection-model/auto_predicted_labels -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
    total_images=$(find detection-model/dataset-multiclass/train/images -name "*.jpg" -type f 2>/dev/null | wc -l | tr -d ' ')
    total_labeled=$((manual_count + auto_count))
    
    if [ $total_images -gt 0 ]; then
        progress=$((total_labeled * 100 / total_images))
        echo "📁 Labels: Manual: $manual_count | Auto: $auto_count | Total: $total_labeled/$total_images ($progress%)"
    else
        echo "📁 Labels: Manual: $manual_count | Auto: $auto_count | Total: $total_labeled"
    fi
    
    # Show progress bar
    if [ $total_images -gt 0 ]; then
        bar_length=50
        filled=$((progress * bar_length / 100))
        empty=$((bar_length - filled))
        bar=$(printf "█%.0s" $(seq 1 $filled))$(printf "░%.0s" $(seq 1 $empty))
        echo "Progress: [$bar] $progress%"
    fi
    
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "📋 Latest Training Log (last 15 lines):"
    echo "─────────────────────────────────────────────────────────────"
    
    if [ -f training_log.txt ]; then
        tail -15 training_log.txt | sed 's/^/  /'
    else
        echo "  No log file found yet..."
    fi
    
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "⏰ Last updated: $(date '+%H:%M:%S.%N' | cut -c1-12)"
    echo "🔄 Live updating... (Ctrl+C to stop)"
    
    sleep 0.1
done
