import streamlit as st
import os
import time

st.title("Lagrange Number Plate System (C++ Backend)")

mode = st.sidebar.selectbox(
    "Mode",
    ["Encode Plate", "Decode Plate"]
)

# ---------------- ENCODE ----------------
if mode == "Encode Plate":

    file = st.file_uploader("Upload plate image")

    if file:

        with open("input.jpg", "wb") as f:
            f.write(file.read())

        st.image("input.jpg")

        if st.button("Encode"):

            os.system("main.exe encode input.jpg")

            time.sleep(1)

            if os.path.exists("key.json"):
                st.success("Encoding complete")

                key_data = open("key.json", "r").read()

                st.subheader("Generated Key")
                st.code(key_data)

                st.info("Copy this key and go to Decode mode")

# ---------------- DECODE ----------------
elif mode == "Decode Plate":

    st.subheader("Paste Lagrange Key")

    # ✅ THIS IS WHAT YOU WERE MISSING
    key_input = st.text_area("Input key here (from encoder)")

    if st.button("Decode"):

        # save user key into file
        with open("key.json", "w") as f:
            f.write(key_input)

        os.system("main.exe decode key.json")

        time.sleep(1)

        if os.path.exists("result.png"):
            st.image("result.png")

        if os.path.exists("result.txt"):
            st.success("Decoded result:")
            st.write(open("result.txt").read())