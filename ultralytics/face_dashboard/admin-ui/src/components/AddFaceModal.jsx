import { Camera, Upload, UserPlus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";

export default function AddFaceModal({ open, onClose, onSuccess }) {
  const [name, setName] = useState("");
  const [faceIndex, setFaceIndex] = useState(0);
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    api.status().then((s) => setCameraOn(s.running)).catch(() => setCameraOn(false));
  }, [open]);

  const reset = () => {
    setName("");
    setFaceIndex(0);
    setPreview(null);
    setFile(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError(null);
  };

  const submitUpload = async () => {
    if (!name.trim()) {
      setError("Enter a name");
      return;
    }
    if (!file) {
      setError("Choose a photo");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await api.registerPerson(name.trim(), file, faceIndex);
      reset();
      onSuccess();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const submitCamera = async () => {
    if (!name.trim()) {
      setError("Enter a name");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await api.registerFromCamera(name.trim(), faceIndex);
      reset();
      onSuccess();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="card w-full max-w-md">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <UserPlus className="h-5 w-5 text-campus-accent" />
            <h3 className="text-lg font-semibold text-white">Add face</h3>
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="rounded-lg p-1 text-slate-400 hover:bg-campus-800 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <label className="mb-3 block text-sm text-slate-400">
          Person name
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Rahul Sharma"
            className="mt-1 w-full rounded-lg border border-campus-700 bg-campus-800 px-3 py-2 text-white outline-none focus:border-campus-accent"
          />
        </label>

        <label className="mb-3 block text-sm text-slate-400">
          Face index (if multiple faces in photo)
          <input
            type="number"
            min={0}
            max={10}
            value={faceIndex}
            onChange={(e) => setFaceIndex(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-campus-700 bg-campus-800 px-3 py-2 text-white"
          />
        </label>

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="mb-3 h-40 w-full rounded-lg object-cover"
          />
        )}

        <div className="mb-4 flex flex-col gap-2 sm:flex-row">
          <button
            type="button"
            className="btn-ghost flex flex-1 items-center justify-center gap-2"
            onClick={() => fileRef.current?.click()}
          >
            <Upload className="h-4 w-4" />
            Upload photo
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onFileChange}
          />
          <button
            type="button"
            className="btn-ghost flex flex-1 items-center justify-center gap-2 disabled:opacity-40"
            onClick={submitCamera}
            disabled={loading || !cameraOn}
            title={cameraOn ? "Capture current camera frame" : "Start camera on Live Camera page first"}
          >
            <Camera className="h-4 w-4" />
            From live camera
          </button>
        </div>

        {!cameraOn && (
          <p className="mb-3 text-xs text-slate-500">
            Start the camera on Live Camera to register from webcam.
          </p>
        )}

        {error && <p className="mb-3 text-sm text-campus-danger">{error}</p>}

        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={handleClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={submitUpload}
            disabled={loading}
          >
            {loading ? "Saving…" : "Save person"}
          </button>
        </div>
      </div>
    </div>
  );
}
