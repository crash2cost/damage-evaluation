#!/bin/bash
# Quick script to check training progress

echo "🔍 OVERNIGHT TRAINING STATUS"
echo "================================"

# Check if process is running
if ps aux | grep -q "[c]ontinuous_training.py"; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show last 20 lines of log
    echo "📊 Latest Log Output:"
    echo "---"
    tail -20 training_log.txt
    echo ""
    
    # Count labels
    manual_count=$(find detection-model/real_labels -name "*.json" -type f | wc -l | tr -d ' ')
    auto_count=$(find detection-model/auto_predicted_labels -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
    
    echo "📁 Label Status:"
    echo "  - Manual labels: $manual_count"
    echo "  - Auto labels: $auto_count"
    echo "  - Total: $((manual_count + auto_count))"
    
else
    echo "⏹️  Training has FINISHED or STOPPED"
    echo ""
    echo "📋 Final Summary:"
    tail -30 training_log.txt
fi

echo ""
echo "💡 Tips:"
echo "  - Full log: cat training_log.txt"
echo "  - Live updates: tail -f training_log.txt"
echo "  - Stop training: pkill -f continuous_training"
