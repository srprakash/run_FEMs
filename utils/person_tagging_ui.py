"""
User interface for tagging persons with identifiers.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PersonTaggingUI:
    """UI for displaying person gallery and collecting identifiers."""
    
    def __init__(self, window_name: str = "Person Tagging"):
        """
        Initialize tagging UI.
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
    def extract_person_images(self, video_frames: Dict[int, np.ndarray], 
                             person_data: Dict) -> List[np.ndarray]:
        """
        Extract sample images of person from video frames using bounding boxes.
        
        Args:
            video_frames: Dictionary mapping frame numbers to frame images
            person_data: Dictionary with person_id, bboxes, etc.
            
        Returns:
            List of person images (cropped from frames)
        """
        person_images = []
        bboxes = person_data.get('bboxes', [])
        
        # Get a few representative images (first, middle, last)
        if not bboxes:
            return person_images
        
        # Sample indices to get diverse images
        num_samples = min(5, len(bboxes))
        if len(bboxes) == 1:
            sample_indices = [0]
        else:
            sample_indices = np.linspace(0, len(bboxes) - 1, num_samples, dtype=int)
        
        # We don't have frame numbers stored, so we'll use the bboxes directly
        # For now, we'll create a simple visualization from the bboxes
        # In a real implementation, you'd want to store frame numbers with bboxes
        
        return person_images
    
    def create_person_gallery(self, person_data_list: List[Dict], 
                             frame_samples: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        """
        Create a gallery image showing all unique persons.
        
        Args:
            person_data_list: List of person data dictionaries
            frame_samples: Optional dictionary mapping person_id to sample frame images
            
        Returns:
            Gallery image with all persons displayed
        """
        if not person_data_list:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Create grid layout
        num_persons = len(person_data_list)
        cols = min(3, num_persons)
        rows = (num_persons + cols - 1) // cols
        
        # Image dimensions per person
        img_width = 200
        img_height = 250
        padding = 10
        
        gallery_width = cols * (img_width + padding) + padding
        gallery_height = rows * (img_height + padding) + padding
        
        gallery = np.ones((gallery_height, gallery_width, 3), dtype=np.uint8) * 255
        
        for idx, person_data in enumerate(person_data_list):
            row = idx // cols
            col = idx % cols
            
            x = col * (img_width + padding) + padding
            y = row * (img_height + padding) + padding
            
            # Create person image placeholder
            person_img = np.ones((img_height - 40, img_width, 3), dtype=np.uint8) * 200
            
            # If we have a sample frame, use it
            if frame_samples and person_data['person_id'] in frame_samples:
                sample_frame = frame_samples[person_data['person_id']]
                # Resize to fit
                h, w = sample_frame.shape[:2]
                aspect = w / h
                if aspect > img_width / (img_height - 40):
                    new_w = img_width
                    new_h = int(img_width / aspect)
                else:
                    new_h = img_height - 40
                    new_w = int((img_height - 40) * aspect)
                
                sample_resized = cv2.resize(sample_frame, (new_w, new_h))
                # Center in person_img
                y_offset = (person_img.shape[0] - new_h) // 2
                x_offset = (person_img.shape[1] - new_w) // 2
                person_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = sample_resized
            else:
                # Draw placeholder text
                cv2.putText(person_img, f"Person {person_data['person_id']}", 
                           (10, (img_height - 40) // 2), 
                           self.font, 0.6, (0, 0, 0), 2)
            
            # Draw border
            cv2.rectangle(person_img, (0, 0), (img_width - 1, img_height - 41), (0, 0, 255), 2)
            
            # Place person image in gallery
            gallery[y:y+img_height-40, x:x+img_width] = person_img
            
            # Draw person ID label
            label = f"ID: {person_data['person_id']}"
            if person_data.get('identifier'):
                label += f" ({person_data['identifier']})"
            
            text_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
            text_x = x + (img_width - text_size[0]) // 2
            text_y = y + img_height - 15
            
            cv2.putText(gallery, label, (text_x, text_y), 
                       self.font, 0.5, (0, 0, 0), 1)
        
        return gallery
    
    
    def tag_persons_interactive(self, person_data_list: List[Dict],
                               frame_samples: Optional[Dict[int, np.ndarray]] = None,
                               allow_temporary_ids: bool = True) -> Dict[int, str]:
        """
        Interactive tagging with individual person display and live tag preview.
        
        Args:
            person_data_list: List of person data dictionaries
            frame_samples: Optional dictionary mapping person_id to sample frame images
            allow_temporary_ids: If True, persons with temporary IDs (like "person_0") are considered untagged
            
        Returns:
            Dictionary mapping person_id to identifier
        """
        if not person_data_list:
            return {}
        
        # Filter out already tagged persons
        # If allow_temporary_ids is True, filter out only persons with real identifiers (not starting with "person_")
        if allow_temporary_ids:
            untagged_persons = [
                p for p in person_data_list 
                if not p.get('identifier') or 
                   p.get('identifier', '').startswith('person_') or 
                   p.get('identifier', '').isdigit()
            ]
        else:
            untagged_persons = [p for p in person_data_list if not p.get('identifier')]
        
        if not untagged_persons:
            print("All persons are already tagged.")
            return {p['person_id']: p.get('identifier') for p in person_data_list if p.get('identifier')}
        
        person_identifiers = {}
        
        print(f"\nFound {len(untagged_persons)} untagged person(s).")
        print("Interactive tagging mode:")
        print("  - Type the identifier and press ENTER to tag")
        print("  - Press ENTER without typing to skip")
        print("  - Type 'q' and press ENTER to quit\n")
        
        window_name = "Person Tagging"
        
        for idx, person_data in enumerate(untagged_persons):
            person_id = person_data['person_id']
            
            # Get person image
            if frame_samples and person_id in frame_samples:
                person_img = frame_samples[person_id].copy()
                # Ensure it's a numpy array and has correct shape
                if not isinstance(person_img, np.ndarray):
                    person_img = np.array(person_img, dtype=np.uint8)
                # Ensure 3 channels if grayscale
                if len(person_img.shape) == 2:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                elif len(person_img.shape) == 3 and person_img.shape[2] == 1:
                    person_img = cv2.cvtColor(person_img, cv2.COLOR_GRAY2BGR)
                print(f"Displaying image for person_id {person_id}: shape {person_img.shape}")
            else:
                # Create placeholder image
                person_img = np.ones((400, 300, 3), dtype=np.uint8) * 200
                cv2.putText(person_img, f"No image for Person {person_id}", (10, 200),
                           self.font, 0.8, (0, 0, 0), 2)
                print(f"WARNING: No image found in frame_samples for person_id {person_id}")
            
            # Resize image for display (make it larger)
            display_height = 600
            aspect = person_img.shape[1] / person_img.shape[0]
            display_width = int(display_height * aspect)
            display_img = cv2.resize(person_img, (display_width, display_height))
            
            # Ensure display_img is BGR format for OpenCV
            if len(display_img.shape) == 2:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            elif display_img.shape[2] == 4:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGRA2BGR)
            elif display_img.shape[2] != 3:
                # Force to 3 channels
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
            # Create display image with border and label area
            border_size = 50
            display_width_with_border = display_width + (border_size * 2)
            display_height_with_border = display_height + (border_size * 3)
            
            # Create the full display canvas (background color: light gray)
            full_display = np.ones((display_height_with_border, display_width_with_border, 3), 
                                  dtype=np.uint8) * 240
            
            # Place person image in center
            y_offset = border_size
            x_offset = border_size
            
            # Place the person image onto the canvas
            full_display[y_offset:y_offset+display_height, x_offset:x_offset+display_width] = display_img
            
            # Draw border around image AFTER placing the image (BGR color: green = (0, 255, 0))
            pt1 = (x_offset-5, y_offset-5)
            pt2 = (x_offset+display_width+5, y_offset+display_height+5)
            cv2.rectangle(full_display, pt1, pt2, (0, 255, 0), 3)
            
            # Draw person info
            info_y = y_offset + display_height + 20
            # Show current identifier if it's a temporary one
            current_id = person_data.get('identifier', f'Person {person_id}')
            if current_id and (current_id.startswith('person_') or current_id.isdigit()):
                person_text = f"{current_id} ({idx + 1}/{len(untagged_persons)})"
            else:
                person_text = f"Person {person_id} ({idx + 1}/{len(untagged_persons)})"
            
            # Use cv2.FONT_HERSHEY_SIMPLEX if font is not set
            font = self.font if hasattr(self, 'font') else cv2.FONT_HERSHEY_SIMPLEX
            
            # Draw person info text
            cv2.putText(full_display, person_text, (x_offset, info_y),
                       font, 0.8, (0, 0, 0), 2)
            
            # Display instruction text
            instruction_y = info_y + 35
            instruction = "Type identifier and press ENTER (or ENTER to skip):"
            cv2.putText(full_display, instruction, (x_offset, instruction_y),
                       font, 0.6, (100, 100, 100), 2)
            
            # Save image to file as well (for debugging and in case display doesn't work)
            try:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                cv2.imwrite(temp_file.name, full_display)
                print(f"Saved person image to: {temp_file.name}")
            except Exception as e:
                print(f"Warning: Could not save image to file: {e}")
            
            # Show window (with error handling)
            try:
                cv2.imshow(window_name, full_display)
                cv2.waitKey(1)  # Small delay to allow window to update
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                print(f"Displaying window '{window_name}'. If window doesn't appear, check saved image file above.")
            except Exception as e:
                print(f"Warning: Could not display OpenCV window: {e}")
                print("Image has been saved to a temporary file (see path above).")
            
            # Get input from user
            print(f"\n[{idx + 1}/{len(untagged_persons)}] Tagging Person {person_id}")
            identifier = input("Enter identifier (name/ID) or press ENTER to skip: ").strip()
            
            if identifier.lower() == 'q':
                print("Tagging cancelled.")
                break
            
            if identifier:
                person_identifiers[person_id] = identifier
                
                # Update display with the tag
                try:
                    tag_y = instruction_y + 35
                    tag_text = f"Tagged: {identifier}"
                    cv2.putText(full_display, tag_text, (x_offset, tag_y),
                               font, 0.7, (0, 255, 0), 2)
                    cv2.imshow(window_name, full_display)
                    cv2.waitKey(500)  # Show confirmation for 0.5 seconds
                except Exception as e:
                    print(f"Warning: Could not update display: {e}")
                print(f"  -> Tagged as: {identifier}")
            else:
                person_identifiers[person_id] = None
                print(f"  -> Skipped")
        
        cv2.destroyWindow(window_name)
        return person_identifiers
    
    def tag_persons(self, person_data_list: List[Dict], 
                   frame_samples: Optional[Dict[int, np.ndarray]] = None,
                   allow_temporary_ids: bool = True) -> Dict[int, str]:
        """
        Display gallery and collect identifiers for each person.
        Uses interactive tagging by default.
        
        Args:
            person_data_list: List of person data dictionaries
            frame_samples: Optional dictionary mapping person_id to sample frame images
            allow_temporary_ids: If True, persons with temporary IDs are considered untagged
            
        Returns:
            Dictionary mapping person_id to identifier (name/ID string)
        """
        # Use interactive tagging by default
        return self.tag_persons_interactive(person_data_list, frame_samples, allow_temporary_ids)

