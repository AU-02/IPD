import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaUserCircle } from "react-icons/fa"; // Profile Icon
import "../themes/theme.css"; 

const Navbar = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const navigate = useNavigate();

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem("token"); // Remove token
    navigate("/"); // Redirect to login
  };

  return (
    <nav className="navbar">
      <div className="nav-left">
        <Link to="/" className="logo">D3-MSD</Link>
      </div>
      <div className="nav-center">
        <Link to="/HomeScreen" className="nav-link">Home</Link>
        <Link to="/tab2" className="nav-link"></Link>
        <Link to="/tab3" className="nav-link"></Link>
      </div>
      <div className="nav-right">
        {/* Profile Icon Clickable */}
        <div className="profile-container">
          <FaUserCircle 
            className="profile-icon" 
            onClick={() => setDropdownOpen(!dropdownOpen)}
          />
          {dropdownOpen && (
            <div className="dropdown">
              <button onClick={handleLogout}>Logout</button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
