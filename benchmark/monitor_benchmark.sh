#!/bin/bash
# Monitor benchmark progress

echo "=== Benchmark Progress Monitor ==="
echo ""

# Check if benchmark is running
if ps aux | grep -q "[b]enchmark_fast_sac_backends"; then
    echo "✓ Benchmark is RUNNING"
    echo ""

    # Show log directories
    echo "Log directories:"
    ls -lt logs/benchmark/ 2>/dev/null | head -5
    echo ""

    # Check tensorboard events
    echo "Latest tensorboard events:"
    find logs/benchmark -name "events.out.tfevents.*" -type f -exec ls -lh {} \; 2>/dev/null | tail -3
    echo ""

    # Show process info
    echo "Process info:"
    ps aux | grep "[b]enchmark_fast_sac_backends" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
else
    echo "✗ Benchmark is NOT running"
    echo ""

    # Check for results
    if [ -f "benchmark/outputs/fast_sac_backends.json" ]; then
        echo "✓ Results file exists:"
        cat benchmark/outputs/fast_sac_backends.json
    fi
fi
