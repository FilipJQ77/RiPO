import cv2 as cv

car_cascade_classifier = cv.CascadeClassifier('cars.xml')


def main():
    while True:
        user_input = input("Wczytać obraz z kamery (1) czy z pliku (2)? ")
        try:
            user_input = int(user_input)
        except Exception as e:
            print(e)
            continue
        if user_input == 1:
            camera_capture = cv.VideoCapture(0)  # 0 = pierwsze urządzenie - kamera
            footage(camera_capture)
        elif user_input == 2:
            filename = input("Podaj nazwę pliku: ")
            file_capture = cv.VideoCapture(filename)
            footage(file_capture)


def footage(capture):
    # sprawdzenie czy capture jest dobrze 'otwarty'
    if not capture.isOpened():
        print("Capture is not opened")
        return

    title = "RiPO"
    cv.namedWindow(title)

    while capture.isOpened():

        car_cascade_classifier = cv.CascadeClassifier('cars.xml')
        # [...]
        # pobranie obrazu z kamery
        reading, frame = capture.read()

        if frame is None:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cars = car_cascade_classifier.detectMultiScale(frame_gray)

        for (x, y, w, h) in cars:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # wyświetlenie obrazu
        cv.imshow(title, frame)

        key = cv.waitKey(1)

        # wyjście z przechwytywania obrazu
        if key == 27:  # ESCAPE
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
