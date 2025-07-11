import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
from smart_assistant import SmartDeskAssistant
import os

# Set page config
st.set_page_config(
    page_title="Smart Desk Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = SmartDeskAssistant()

def main():
    st.title("🧠 Smart Desk Assistant")
    st.write("Upload an image of your desk to detect and classify items!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a desk image...",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save uploaded file temporarily
        temp_path = "temp_desk_image.jpg"
        image.save(temp_path)
        
        # Add analysis button
        if st.button("Analyze Desk"):
            with st.spinner("Analyzing your desk..."):
                # Analyze the image
                results = st.session_state.assistant.analyze_desk(temp_path)
                
                # Display results
                st.success(f"Analysis complete! Found {results['total_objects']} objects.")
                
                # Show detailed results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Detected Items")
                    for item in results['items']:
                        st.write(f"**{item['detected_class']}** - Confidence: {item['classification_confidence']:.2f}")
                
                with col2:
                    st.subheader("Item Summary")
                    item_counts = {}
                    for item in results['items']:
                        class_name = item['detected_class']
                        item_counts[class_name] = item_counts.get(class_name, 0) + 1
                    
                    for item_type, count in item_counts.items():
                        st.write(f"{item_type}: {count}")
                
                # Create and display annotated image
                st.session_state.assistant.create_annotated_image(temp_path, results, "annotated_result.jpg")
                
                if os.path.exists("annotated_result.jpg"):
                    annotated_image = Image.open("annotated_result.jpg")
                    st.image(annotated_image, caption="Detected Objects", use_column_width=True)
               
                # Show raw results
                with st.expander("View Raw Results"):
                    st.json(results)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
