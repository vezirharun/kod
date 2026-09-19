from core.face_identity import FaceIdentityEngine

if __name__ == '__main__':
    e = FaceIdentityEngine()
    print('Face Intelligence V10')
    print('backend:', e.backend)
    print('capability:', e.capability())
    print('available:', e.available())
    print('effective_threshold:', e.effective_threshold)
