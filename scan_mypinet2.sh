#!/bin/bash

# Script to scan MyPiNet2 network and restore original connection
OUTPUT_FILE="mypinet2_devices.txt"
WIFI_INTERFACE="en0"
WIFI_SERVICE="Wi-Fi"

echo "Starting network scan script..."
echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Save current network state
echo "Saving current network state..."
CURRENT_NETWORK=$(networksetup -getairportnetwork $WIFI_INTERFACE 2>&1)
CURRENT_IP=$(ifconfig $WIFI_INTERFACE | grep "inet " | awk '{print $2}')
echo "Current network: $CURRENT_NETWORK"
echo "Current IP: $CURRENT_IP"
echo ""

# Prompt for MyPiNet2 password
echo -n "Enter password for MyPiNet2: "
read -s MYPINET2_PASSWORD
echo ""

# Connect to MyPiNet2
echo "Connecting to MyPiNet2..."
networksetup -setairportnetwork $WIFI_INTERFACE MyPiNet2 "$MYPINET2_PASSWORD"

if [ $? -ne 0 ]; then
    echo "Failed to connect to MyPiNet2"
    exit 1
fi

# Wait for connection to establish
echo "Waiting for connection to establish..."
sleep 5

# Verify connection
NEW_NETWORK=$(networksetup -getairportnetwork $WIFI_INTERFACE 2>&1)
echo "Connected to: $NEW_NETWORK"

# Get new IP address
NEW_IP=$(ifconfig $WIFI_INTERFACE | grep "inet " | awk '{print $2}')
echo "New IP address: $NEW_IP"
echo ""

# Scan for devices
echo "Scanning for devices on MyPiNet2..."
echo "=====================================" > "$OUTPUT_FILE"
echo "MyPiNet2 Network Scan" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "=====================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Get subnet from IP address
if [ -n "$NEW_IP" ]; then
    SUBNET=$(echo $NEW_IP | cut -d'.' -f1-3)
    echo "Scanning subnet: ${SUBNET}.0/24" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Use arp-scan if available, otherwise use arp + ping sweep
    if command -v arp-scan &> /dev/null; then
        echo "Using arp-scan..." | tee -a "$OUTPUT_FILE"
        sudo arp-scan --interface=$WIFI_INTERFACE ${SUBNET}.0/24 >> "$OUTPUT_FILE"
    else
        echo "Performing ping sweep and checking ARP table..." | tee -a "$OUTPUT_FILE"
        
        # Ping sweep (in background)
        for i in {1..254}; do
            ping -c 1 -t 1 ${SUBNET}.$i &> /dev/null &
        done
        
        # Wait for pings to complete
        sleep 10
        
        # Get ARP table
        echo "" >> "$OUTPUT_FILE"
        echo "ARP Table:" >> "$OUTPUT_FILE"
        arp -a >> "$OUTPUT_FILE"
    fi
else
    echo "ERROR: Could not determine IP address on MyPiNet2" | tee -a "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"
echo "Scan complete!" | tee -a "$OUTPUT_FILE"
echo ""

# Display results
echo "======================================"
echo "Devices found:"
echo "======================================"
cat "$OUTPUT_FILE"
echo ""

# Restore original connection
echo "Restoring original network connection..."

# Check if we need to reconnect to a specific network
if [[ "$CURRENT_NETWORK" == *"Current Wi-Fi Network:"* ]]; then
    # Extract network name
    RESTORE_NETWORK=$(echo "$CURRENT_NETWORK" | sed 's/Current Wi-Fi Network: //')
    echo "Reconnecting to: $RESTORE_NETWORK"
    
    # Prompt for original network password if needed
    echo -n "Enter password for $RESTORE_NETWORK (press Enter if saved): "
    read -s RESTORE_PASSWORD
    echo ""
    
    if [ -n "$RESTORE_PASSWORD" ]; then
        networksetup -setairportnetwork $WIFI_INTERFACE "$RESTORE_NETWORK" "$RESTORE_PASSWORD"
    else
        networksetup -setairportnetwork $WIFI_INTERFACE "$RESTORE_NETWORK"
    fi
    
    sleep 3
    RESTORED=$(networksetup -getairportnetwork $WIFI_INTERFACE 2>&1)
    echo "Restored to: $RESTORED"
else
    echo "No WiFi network to restore (you may have been on Ethernet)"
fi

echo ""
echo "======================================"
echo "Script complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "======================================"
