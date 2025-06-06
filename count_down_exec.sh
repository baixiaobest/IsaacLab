#!/bin/bash

# Show help function
show_help() {
    echo "Countdown Timer with Command Execution"
    echo "Usage: $0 [DURATION] \"[COMMAND]\""
    echo "       $0 -h | --help"
    echo ""
    echo "DURATION Formats:"
    echo "  HH:MM:SS    Hours:Minutes:Seconds (e.g., 02:30:00)"
    echo "  MM:SS       Minutes:Seconds (e.g., 10:00 or 90:00 = 1hr 30min)"
    echo "  SS          Seconds only (e.g., 3600 = 1hr)"
    echo ""
    echo "COMMAND: Any shell command to execute after countdown (must be quoted)"
    echo ""
    echo "Examples:"
    echo "  $0 00:05:00  \"notify-send 'Timer' 'Pizza is ready!'\""
    echo "  $0 10:00     'echo \"Break time!\" && paplay /usr/share/sounds/alarm.oga'"
    echo "  $0 3600      \"curl -X POST http://example.com/api/timer\""
    echo "  $0 90:00     \"shutdown -h now\"   # 90 minutes = 1.5 hours"
    echo ""
    echo "Notes:"
    echo "  - Always quote commands containing spaces or special characters"
    echo "  - Supports minutes values beyond 59 (automatically converted to hours)"
    echo "  - Displays countdown in normalized HH:MM:SS format"
    echo "  - Press Ctrl+C to interrupt the timer"
}

# Check for help request
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Validate arguments
if [[ $# -lt 2 ]]; then
    echo "ERROR: Missing arguments"
    show_help
    exit 1
fi

# Parse time input
time_input="$1"
shift
end_command="$*"

# Convert time to seconds based on format
if [[ "$time_input" =~ ^[0-9]+:[0-9]+:[0-9]+$ ]]; then
    # HH:MM:SS format
    IFS=":" read -r hours minutes seconds <<< "$time_input"
    total_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
elif [[ "$time_input" =~ ^[0-9]+:[0-9]+$ ]]; then
    # MM:SS format
    IFS=":" read -r minutes seconds <<< "$time_input"
    total_seconds=$((10#$minutes * 60 + 10#$seconds))
elif [[ "$time_input" =~ ^[0-9]+$ ]]; then
    # SS format
    total_seconds=$((10#$time_input))
else
    echo "ERROR: Invalid time format"
    show_help
    exit 1
fi

# Validate total seconds
if [[ $total_seconds -le 0 ]]; then
    echo "ERROR: Time must be positive"
    show_help
    exit 1
fi

# Countdown function
countdown() {
    local remaining=$1
    while [[ $remaining -gt 0 ]]; do
        # Calculate time components
        local hrs=$((remaining / 3600))
        local mins=$(((remaining % 3600) / 60))
        local secs=$((remaining % 60))
        
        # Format with leading zeros
        printf "\r%02d:%02d:%02d" "$hrs" "$mins" "$secs"
        sleep 1
        ((remaining--))
    done
    printf "\r%02d:%02d:%02d\n" 0 0 0
}

# Run countdown
echo "Starting timer for: $time_input (${total_seconds}s) to execute $end_command"
countdown "$total_seconds"

# Execute command after countdown
echo "Executing: $end_command"
eval "$end_command"
