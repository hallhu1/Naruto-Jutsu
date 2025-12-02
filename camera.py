import cv2
from jutsu_matcher import load_groundtruths, find_jutsu

def start_camera():
    """
    Start live camera feed processing
    """
    gt = load_groundtruths()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        
        # Show the frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        display_msg = "No Jutsu detected"
        try:
            closest, distance = find_jutsu(gt, frame)
            if distance < 7:
                # Draw top banner
                display_msg = closest

        except ValueError:
            pass

        cv2.putText(frame, display_msg, (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                            (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Camera Feed', frame)


        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()