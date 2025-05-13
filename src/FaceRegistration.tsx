import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import * as faceapi from 'face-api.js';

const detectionOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 128,
    scoreThreshold: 0.3,
});

type UserFace = {
    id: string;
    descriptor: Float32Array;
};

const FaceRegistration: React.FC = () => {
    const webcamRef = useRef<Webcam>(null);
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [message, setMessage] = useState('');
    const [users, setUsers] = useState<UserFace[]>([]);
    const [userCount, setUserCount] = useState(1);

    useEffect(() => {
        const loadModels = async () => {
            try {
                await Promise.all([
                    faceapi.nets.tinyFaceDetector.loadFromUri('/models/tiny_face_detector'),
                    faceapi.nets.faceLandmark68Net.loadFromUri('/models/face_landmark_68'),
                    faceapi.nets.faceRecognitionNet.loadFromUri('/models/face_recognition'),
                ]);
                setModelsLoaded(true);
                setMessage('Модели загружены. Можете регистрировать лицо.');
            } catch (e) {
                setMessage('Ошибка загрузки моделей');
                console.error(e);
            }
        };
        loadModels();
    }, []);

    const registerFace = async () => {
        if (!modelsLoaded) {
            setMessage('Модели еще не загружены!');
            return;
        }
        const video = webcamRef.current?.video;
        if (!video) {
            setMessage('Нет доступа к камере.');
            return;
        }

        setMessage('Идет поиск лица для регистрации...');
        const detection = await faceapi
            .detectSingleFace(video, detectionOptions)
            .withFaceLandmarks()
            .withFaceDescriptor();

        if (detection && detection.descriptor) {
            const newUser: UserFace = {
                id: `user${userCount}`,
                descriptor: detection.descriptor,
            };
            setUsers((prev) => [...prev, newUser]);
            setUserCount((count) => count + 1);
            setMessage(`Лицо зарегистрировано как ${newUser.id}`);
            console.log('Зарегистрированные пользователи:', [...users, newUser]);
        } else {
            setMessage('Лицо не найдено. Попробуйте еще раз.');
        }
    };

    const recognizeFace = async () => {
        if (!modelsLoaded) {
            setMessage('Модели еще не загружены!');
            return;
        }
        if (users.length === 0) {
            setMessage('Нет зарегистрированных пользователей для сравнения.');
            return;
        }
        const video = webcamRef.current?.video;
        if (!video) {
            setMessage('Нет доступа к камере.');
            return;
        }

        setMessage('Идет поиск лица для распознавания...');
        const detection = await faceapi
            .detectSingleFace(video, detectionOptions)
            .withFaceLandmarks()
            .withFaceDescriptor();

        if (detection && detection.descriptor) {
            const queryDescriptor = detection.descriptor;

            // Используем reduce для поиска лучшего совпадения
            const bestMatch = users.reduce<{ id: string; distance: number } | null>((best, user) => {
                const distance = faceapi.euclideanDistance(queryDescriptor, user.descriptor);
                if (best === null || distance < best.distance) {
                    return { id: user.id, distance };
                }
                return best;
            }, null);

            if (bestMatch !== null && bestMatch.distance < 0.6) {
                setMessage(`Распознано лицо: ${bestMatch.id} (расстояние ${bestMatch.distance.toFixed(3)})`);
            } else {
                setMessage('Лицо не распознано.');
            }
        } else {
            setMessage('Лицо не найдено. Попробуйте еще раз.');
        }
    };

    return (
        <div style={{ textAlign: 'center' }}>
            <h2>Регистрация и распознавание лица</h2>
            <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                width={320}
                height={240}
                videoConstraints={{ facingMode: 'user', width: 320, height: 240 }}
            />
            <br />
            <button onClick={registerFace} disabled={!modelsLoaded} style={{ margin: '10px' }}>
                Зарегистрировать лицо
            </button>
            <button onClick={recognizeFace} disabled={!modelsLoaded || users.length === 0} style={{ margin: '10px' }}>
                Распознать лицо
            </button>
            <div style={{ marginTop: 20, color: '#555' }}>{message}</div>
            <div style={{ marginTop: 10 }}>
                <b>Зарегистрированные пользователи:</b>
                <ul>
                    {users.map((user) => (
                        <li key={user.id}>{user.id}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default FaceRegistration;
