# ============================================================
# PHONE DETECTION & MONITORING SYSTEM
# ============================================================
# Model Information
# Model: YOLOv8 Nano
# Task: Object Detection
# Class: Phone
# Tracker: ByteTrack
# ============================================================


import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Phone Monitoring System",
    page_icon="📱",
    layout="wide"
)


# ==============================
# TITLE
# ==============================

st.title("📱 Phone Detection & Monitoring System")

st.write(
    "Upload a video to detect, track and monitor mobile phones."
)


# ==============================
# LOAD MODEL
# ==============================

MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)


# ==============================
# SIDEBAR SETTINGS
# ==============================

st.sidebar.header("⚙️ Monitoring Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

alert_threshold = st.sidebar.slider(
    "Alert Threshold (seconds)",
    min_value=1,
    max_value=10,
    value=3
)


# ==============================
# VIDEO UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ==============================
# PROCESS VIDEO
# ==============================

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🚀 Start Monitoring"):

        # ------------------------------
        # Save uploaded video temporarily
        # ------------------------------

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(
            uploaded_file.read()
        )

        temp_video.close()

        video_path = temp_video.name


        # ------------------------------
        # Read video information
        # ------------------------------

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        cap.release()


        # ------------------------------
        # Output video
        # ------------------------------

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        # ==============================
        # YOLO TRACKING
        # ==============================

        results = model.track(
            source=video_path,
            tracker="bytetrack.yaml",
            conf=confidence,
            imgsz=640,
            stream=True
        )


        # ==============================
        # TRACKING VARIABLES
        # ==============================

        phone_data = {}

        events = []

        frame_number = 0

        MAX_MISSING_FRAMES = 5

        MIN_DURATION = 0.5


        # ==============================
        # STREAMLIT PROGRESS
        # ==============================

        progress = st.progress(0)

        status_text = st.empty()


        # ==============================
        # PROCESS EACH FRAME
        # ==============================

        for result in results:

            frame_number += 1

            current_time = (
                frame_number - 1
            ) / fps


            # ------------------------------
            # Get current tracking IDs
            # ------------------------------

            current_ids = set()


            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )

                current_ids.update(track_ids)


                # ------------------------------
                # Update phone tracking
                # ------------------------------

                for phone_id, box in zip(
                    track_ids,
                    boxes
                ):

                    if phone_id not in phone_data:

                        phone_data[phone_id] = {

                            "start_frame":
                                frame_number,

                            "last_seen_frame":
                                frame_number

                        }

                    else:

                        phone_data[
                            phone_id
                        ][
                            "last_seen_frame"
                        ] = frame_number


            # ------------------------------
            # Check disappeared phones
            # ------------------------------

            for phone_id in list(phone_data.keys()):

                if phone_id not in current_ids:

                    missing_frames = (
                        frame_number
                        -
                        phone_data[
                            phone_id
                        ][
                            "last_seen_frame"
                        ]
                    )


                    if missing_frames > MAX_MISSING_FRAMES:

                        start_frame = (
                            phone_data[
                                phone_id
                            ][
                                "start_frame"
                            ]
                        )

                        end_frame = (
                            phone_data[
                                phone_id
                            ][
                                "last_seen_frame"
                            ]
                        )


                        start_time = (
                            start_frame - 1
                        ) / fps

                        end_time = (
                            end_frame
                        ) / fps

                        duration = (
                            end_time
                            -
                            start_time
                        )


                        # Only save meaningful events

                        if duration >= MIN_DURATION:

                            events.append({

                                "Phone ID":
                                    phone_id,

                                "Start Time (sec)":
                                    round(
                                        start_time,
                                        2
                                    ),

                                "End Time (sec)":
                                    round(
                                        end_time,
                                        2
                                    ),

                                "Duration (sec)":
                                    round(
                                        duration,
                                        2
                                    )

                            })


                        del phone_data[
                            phone_id
                        ]


            # ==============================
            # DRAW DETECTION
            # ==============================

            frame = result.plot()

            phone_count = len(
                current_ids
            )


            # ==============================
            # DRAW PHONE INFORMATION
            # ==============================

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )


                for phone_id, box in zip(
                    track_ids,
                    boxes
                ):

                    start_frame = (
                        phone_data[
                            phone_id
                        ][
                            "start_frame"
                        ]
                    )

                    duration = (
                        frame_number
                        -
                        start_frame
                        +
                        1
                    ) / fps


                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )


                    # ------------------------------
                    # Alert
                    # ------------------------------

                    if duration >= alert_threshold:

                        status = (
                            "WARNING: PHONE "
                            "DETECTED TOO LONG"
                        )

                        text_color = (
                            0,
                            0,
                            255
                        )

                    else:

                        status = "Phone detected"

                        text_color = (
                            255,
                            255,
                            255
                        )


                    # ------------------------------
                    # ID + Duration
                    # ------------------------------

                    cv2.putText(

                        frame,

                        f"ID: {phone_id} | "
                        f"{duration:.2f}s",

                        (
                            x1,
                            max(
                                y1 - 35,
                                30
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 255, 0),

                        2

                    )


                    # ------------------------------
                    # Status
                    # ------------------------------

                    cv2.putText(

                        frame,

                        status,

                        (
                            x1,
                            max(
                                y1 - 10,
                                55
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.6,

                        text_color,

                        2

                    )


            # ==============================
            # GENERAL VIDEO INFORMATION
            # ==============================

            cv2.putText(

                frame,

                f"Phones Detected: {phone_count}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f"Video Time: {current_time:.2f}s",

                (20, 75),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2

            )


            # ------------------------------
            # Save frame
            # ------------------------------

            out.write(frame)


            # ------------------------------
            # Update progress
            # ------------------------------

            if total_frames > 0:

                progress_value = (
                    frame_number
                    /
                    total_frames
                )

                progress.progress(
                    min(
                        progress_value,
                        1.0
                    )
                )


            status_text.text(

                f"Processing frame "
                f"{frame_number} / "
                f"{total_frames}"

            )


        # ==============================
        # SAVE REMAINING ACTIVE PHONES
        # ==============================

        for phone_id in list(
            phone_data.keys()
        ):

            start_frame = (
                phone_data[
                    phone_id
                ][
                    "start_frame"
                ]
            )

            end_frame = (
                phone_data[
                    phone_id
                ][
                    "last_seen_frame"
                ]
            )


            start_time = (
                start_frame - 1
            ) / fps

            end_time = (
                end_frame
            ) / fps

            duration = (
                end_time
                -
                start_time
            )


            if duration >= MIN_DURATION:

                events.append({

                    "Phone ID":
                        phone_id,

                    "Start Time (sec)":
                        round(
                            start_time,
                            2
                        ),

                    "End Time (sec)":
                        round(
                            end_time,
                            2
                        ),

                    "Duration (sec)":
                        round(
                            duration,
                            2
                        )

                })


        # ==============================
        # FINISH VIDEO
        # ==============================

        out.release()

        progress.progress(1.0)

        status_text.success(
            "Monitoring completed! ✅"
        )


        # ==============================
        # CREATE DATAFRAME
        # ==============================

        df = pd.DataFrame(
            events
        )


        # ==============================
        # DASHBOARD METRICS
        # ==============================

        st.subheader(
            "📊 Monitoring Summary"
        )


        if len(df) > 0:

            total_events = len(df)

            max_duration = df[
                "Duration (sec)"
            ].max()

            warning_events = len(
                df[
                    df[
                        "Duration (sec)"
                    ]
                    >=
                    alert_threshold
                ]
            )

        else:

            total_events = 0

            max_duration = 0

            warning_events = 0


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "📱 Total Phone Events",
            total_events
        )


        col2.metric(
            "⏱️ Maximum Duration",
            f"{max_duration:.2f} sec"
        )


        col3.metric(
            "⚠️ Long-Duration Events",
            warning_events
        )


        # ==============================
        # PROCESSED VIDEO
        # ==============================

        st.subheader(
            "🎥 Processed Video"
        )


        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = (
                video_file.read()
            )


        st.video(
            video_bytes
        )


        st.download_button(

            label=
                "⬇️ Download Processed Video",

            data=video_bytes,

            file_name=
                "phone_monitoring.mp4",

            mime=
                "video/mp4"

        )


        # ==============================
        # EVENT LOG
        # ==============================

        st.subheader(
            "📋 Phone Monitoring Log"
        )


        if len(df) > 0:

            st.dataframe(
                df,
                use_container_width=True
            )


            # ------------------------------
            # CSV
            # ------------------------------

            csv_data = df.to_csv(
                index=False
            )


            st.download_button(

                label=
                    "📥 Download CSV Report",

                data=csv_data,

                file_name=
                    "phone_monitoring_log.csv",

                mime=
                    "text/csv"

            )

        else:

            st.info(
                "No phone events were recorded."
            )


        # ==============================
        # CLEAN TEMP FILE
        # ==============================

        os.remove(
            video_path
        )