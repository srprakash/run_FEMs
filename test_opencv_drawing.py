#!/usr/bin/env python3
"""
Simple test program to debug OpenCV drawing functions.
"""

import cv2
import numpy as np
import sys

def test_opencv_drawing():
    """Test basic OpenCV drawing functions."""
    print("Creating test image...")
    
    # Create a simple image
    width = 800
    height = 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Test 1: Draw a rectangle
    print("\nTest 1: Drawing rectangle...")
    try:
        pt1 = (50, 50)
        pt2 = (300, 200)
        color = (0, 255, 0)  # Green in BGR
        thickness = 3
        cv2.rectangle(img, pt1, pt2, color, thickness)
        print(f"  ✓ Rectangle drawn from {pt1} to {pt2}")
    except Exception as e:
        print(f"  ✗ Error drawing rectangle: {e}")
        return False
    
    # Test 2: Draw filled rectangle
    print("\nTest 2: Drawing filled rectangle...")
    try:
        pt1 = (350, 50)
        pt2 = (600, 200)
        color = (255, 0, 0)  # Blue in BGR
        cv2.rectangle(img, pt1, pt2, color, -1)  # -1 = filled
        print(f"  ✓ Filled rectangle drawn from {pt1} to {pt2}")
    except Exception as e:
        print(f"  ✗ Error drawing filled rectangle: {e}")
        return False
    
    # Test 3: Draw text
    print("\nTest 3: Drawing text...")
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Hello OpenCV!"
        position = (50, 300)
        font_scale = 1.0
        color = (0, 0, 0)  # Black
        thickness = 2
        
        cv2.putText(img, text, position, font, font_scale, color, thickness)
        print(f"  ✓ Text '{text}' drawn at {position}")
    except Exception as e:
        print(f"  ✗ Error drawing text: {e}")
        return False
    
    # Test 4: Draw multiple lines of text
    print("\nTest 4: Drawing multiple lines of text...")
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            "Line 1: Rectangle test",
            "Line 2: Text test",
            "Line 3: All tests passed!"
        ]
        y_pos = 350
        for i, line in enumerate(lines):
            pos = (50, y_pos + i * 30)
            cv2.putText(img, line, pos, font, 0.6, (100, 100, 100), 2)
            print(f"  ✓ Line {i+1} drawn at {pos}")
    except Exception as e:
        print(f"  ✗ Error drawing multiple lines: {e}")
        return False
    
    # Test 5: Draw circle
    print("\nTest 5: Drawing circle...")
    try:
        center = (700, 300)
        radius = 50
        color = (0, 0, 255)  # Red
        thickness = 3
        cv2.circle(img, center, radius, color, thickness)
        print(f"  ✓ Circle drawn at {center} with radius {radius}")
    except Exception as e:
        print(f"  ✗ Error drawing circle: {e}")
        return False
    
    # Save image to file
    output_file = "test_opencv_drawing.png"
    print(f"\nSaving image to: {output_file}")
    try:
        success = cv2.imwrite(output_file, img)
        if success:
            print(f"  ✓ Image saved successfully to {output_file}")
        else:
            print(f"  ✗ Failed to save image to {output_file}")
    except Exception as e:
        print(f"  ✗ Error saving image: {e}")
    
    # Display image
    window_name = "OpenCV Drawing Test"
    print(f"\nDisplaying window: {window_name}")
    print("Press any key to close the window...")
    
    try:
        cv2.imshow(window_name, img)
        print(f"  ✓ Window created and image displayed")
        
        # Wait for key press
        key = cv2.waitKey(0)
        print(f"  ✓ Key pressed: {key}")
        
        cv2.destroyWindow(window_name)
        print(f"  ✓ Window destroyed")
        
    except Exception as e:
        print(f"  ✗ Error displaying window: {e}")
        print(f"  Check the saved image file '{output_file}' instead")
        return False
    
    print("\n" + "="*50)
    print("All tests completed!")
    print(f"Check the saved image: {output_file}")
    return True

if __name__ == "__main__":
    print("="*50)
    print("OpenCV Drawing Function Test")
    print("="*50)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")
    print()
    
    success = test_opencv_drawing()
    
    if success:
        print("\n✓ All drawing functions appear to be working!")
        sys.exit(0)
    else:
        print("\n✗ Some drawing functions failed. Check errors above.")
        sys.exit(1)


