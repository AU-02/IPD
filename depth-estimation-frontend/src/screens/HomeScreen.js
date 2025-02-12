import React, { useEffect, useState } from "react";

const HomeScreen = () => {
    const [user, setUser] = useState(null);
    const [error, setError] = useState("");
    const [image, setImage] = useState(null);
    const [depthMap, setDepthMap] = useState(null);

    useEffect(() => {
        const token = localStorage.getItem("token");
        if (!token) {
            window.location.href = "/";
            return;
        }

        fetch("http://127.0.0.1:8000/home", {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Unauthorized or session expired");
            }
            return response.json();
        })
        .then(data => setUser(data))
        .catch(() => {
            setError("Session expired. Please log in again.");
            localStorage.removeItem("token");
            window.location.href = "/";
        });
    }, []);

    // Handle Image Upload
    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
    
        setImage(URL.createObjectURL(file));
    
        const formData = new FormData();
        formData.append("file", file);
    
        const token = localStorage.getItem("token"); //  Retrieve token
    
        fetch("http://127.0.0.1:8000/depth/depth-map", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`, // Add Authorization Header
            },
            body: formData,
        })
        .then(response => response.json()) // Expect JSON response with depth_map_url
        .then(data => {
            if (data.depth_map_url) {
                console.log("🔍 Depth Map URL:", data.depth_map_url);
                setDepthMap(data.depth_map_url); // Use backend depth map URL
            } else {
                console.error("No depth_map_url received from backend.");
            }
        })
        .catch(error => console.error("Error processing depth:", error));
    };

    return (
        <div className="home-container">
            {/* Left Section - Title, Description & Upload */}
            <div className="left-home-section">
                <h1 className="title">Input Image</h1>
                <p className="description">
                    Need Depth Map Generated? Insert a NIR/Thermal Image and get instant output!
                </p>
                <input id="file-upload" type="file" className="file-input" onChange={handleImageUpload} />
                <label htmlFor="file-upload" className="upload-btn">Insert</label>
            </div>

           {/* Right Section - Depth Map Display */}
            <div className="right-home-section">
                <div className="depth-box">
                    {depthMap ? (
                        <img src={depthMap} alt="Depth Map" className="depth-image" />
                    ) : (
                        <p className="placeholder-text">Depth map will appear here.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default HomeScreen;
