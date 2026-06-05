import os
import wave
import struct
import tempfile
from fastapi.testclient import TestClient
from backend.main import app

def create_dummy_wav():
    """Generates a short, valid 16kHz mono PCM WAV file in temp space."""
    temp_dir = tempfile.gettempdir()
    temp_wav = os.path.join(temp_dir, "dummy_verify.wav")
    
    # 16000 Hz, 16-bit, 1 channel, 1-second sine wave
    sample_rate = 16000
    duration = 1.0  
    num_samples = int(sample_rate * duration)
    
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Write dummy silent samples (zero amplitude)
        for _ in range(num_samples):
            data = struct.pack('<h', 0)
            wav_file.writeframesraw(data)
            
    return temp_wav

def verify_system_endpoints():
    print("=== STARTING ENDPOINT VERIFICATION TESTS ===")
    
    client = TestClient(app)
    
    # Create dummy wav file
    wav_path = create_dummy_wav()
    print(f"1. Created temporary verification audio: {wav_path}")
    
    try:
        # Step 1: Upload endpoint
        print("\n2. Testing POST /api/upload...")
        with open(wav_path, 'rb') as f:
            response = client.post(
                "/api/upload",
                files={"file": ("dummy.wav", f, "audio/wav")},
                data={"patient_id": "TEST-PATIENT-999"}
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        res_data = response.json()
        assert "session_id" in res_data
        session_id = res_data["session_id"]
        server_audio_path = res_data["audio_path"]
        print(f"   [SUCCESS] Session: {session_id}, Path: {server_audio_path}")
        
        # Step 2: Preprocess endpoint
        print("\n3. Testing POST /api/preprocess...")
        response = client.post(
            "/api/preprocess",
            json={"audio_path": server_audio_path, "patient_id": "TEST-PATIENT-999"}
        )
        assert response.status_code == 200, f"Preprocessing failed: {response.text}"
        res_data = response.json()
        assert res_data["segments_count"] > 0
        segments_count = res_data["segments_count"]
        print(f"   [SUCCESS] Preprocessing completed. Segment count: {segments_count}")
        
        # Step 3: Feature extraction
        print("\n4. Testing POST /api/extract...")
        response = client.post(
            "/api/extract",
            json={
                "session_id": session_id,
                "audio_path": server_audio_path,
                "segments_count": segments_count
            }
        )
        assert response.status_code == 200, f"Feature extraction failed: {response.text}"
        res_data = response.json()
        assert "ssi_base_score" in res_data
        segment_features = res_data["segment_features"]
        print(f"   [SUCCESS] Features extracted. Base SSI Score: {res_data['ssi_base_score']:.4f}")
        
        # Step 4: Latent representation learning
        print("\n5. Testing POST /api/latent...")
        response = client.post(
            "/api/latent",
            json={
                "session_id": session_id,
                "features": segment_features
            }
        )
        assert response.status_code == 200, f"Latent learning failed: {response.text}"
        res_data = response.json()
        assert "vq_dimensions" in res_data
        print(f"   [SUCCESS] Latent representations generated.")
        
        # Step 5: Attention Fusion
        print("\n6. Testing POST /api/fuse...")
        response = client.post(
            "/api/fuse",
            json={
                "session_id": session_id,
                "latent_reps": []
            }
        )
        assert response.status_code == 200, f"Attention fusion failed: {response.text}"
        res_data = response.json()
        assert res_data["fused_count"] > 0
        print(f"   [SUCCESS] Attention fusion completed.")
        
        # Step 6: MLP Regression Prediction
        print("\n7. Testing POST /api/predict...")
        response = client.post(
            "/api/predict",
            json={
                "session_id": session_id,
                "fused_features": {}
            }
        )
        assert response.status_code == 200, f"Regression prediction failed: {response.text}"
        res_data = response.json()
        assert "ssi_score" in res_data
        assert "panss_score" in res_data
        print(f"   [SUCCESS] SSI Score: {res_data['ssi_score']:.4f}, PANSS: {res_data['panss_score']:.1f}, Severity: {res_data['severity']}")
        
        # History check
        print("\n8. Testing GET /api/history...")
        response = client.get("/api/history")
        assert response.status_code == 200
        records = response.json()["records"]
        assert len(records) > 0
        print(f"   [SUCCESS] Found {len(records)} history entries.")
        
        # Metrics check
        print("\n9. Testing GET /api/metrics...")
        response = client.get("/api/metrics")
        assert response.status_code == 200
        metrics = response.json()["metrics"]
        assert "mae" in metrics
        print(f"   [SUCCESS] Evaluation MAE: {metrics['mae']}")
        
        print("\n=== ALL SYSTEM ENDPOINTS VERIFIED SUCCESSFULLY ===")
        return True
        
    except AssertionError as e:
        print(f"\n[FAILURE] Verification check failed: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILURE] Exception during verification: {e}")
        return False
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    verify_system_endpoints()
