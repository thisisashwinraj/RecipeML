import re
import random

import time
from PIL import Image
import pandas as pd
import streamlit as st

from image_generation import GenerativeImageSynthesis


if __name__ == "__main__":
    user_input_query = st.sidebar.text_input("Enter the name of a cuisine")

    if user_input_query:
        st.markdown(f"<H2>Showing Results for {user_input_query.title()}</H2>", unsafe_allow_html=True)
        st.markdown("<P align='justify'>These images are created using Generative AI. While we strive for accuracy and realism, AI-generated images may not always be perfect & could contain inconsistencies, and/or inaccuracies. Caution is advised</P>", unsafe_allow_html=True)
        st.write(" ")

        genesis_with_gpu = GenerativeImageSynthesis(enable_gpu_acceleration=True)
        genesis_with_cpu = GenerativeImageSynthesis(enable_gpu_acceleration=False)

        image_section_1, image_section_2, image_section_3 = st.columns(3)

        with image_section_1:
            start_time = time.time()
            standard_quality_image = genesis_with_gpu.generate_image(user_input_query, 225, 225, 'standard')

            if standard_quality_image:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open(standard_quality_image), caption=f'Image using RunwayML ({elapsed_time:.1f} secs)')            
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open('assets\images\placeholder\placeholder_2.png'), caption=f'Placeholder Image ({elapsed_time:.1f} secs)')

        with image_section_2:
            start_time = time.time()
            low_quality_image = genesis_with_cpu.generate_image(user_input_query, 225, 225, 'low')

            if low_quality_image:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open(low_quality_image), caption=f'Image using DALL.E2 ({elapsed_time:.1f} secs)')
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open('assets\images\placeholder\placeholder_3.png'), caption=f'Placeholder Image ({elapsed_time:.1f} secs)')

        with image_section_3:
            start_time = time.time()
            high_quality_image = genesis_with_gpu.generate_image(user_input_query, 225, 225, 'high')
            
            if high_quality_image:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open(high_quality_image), caption=f'Image using PlaygroundAI ({elapsed_time:.1f} secs)')
            else:
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.image(Image.open('assets\images\placeholder\placeholder_1.png'), caption=f'Placeholder Image ({elapsed_time:.1f} secs)')
