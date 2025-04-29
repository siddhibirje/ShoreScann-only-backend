import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const navigate = useNavigate();
  const getInvolvedRef = useRef(null);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToAbout = () => {
    if (getInvolvedRef.current) {
      getInvolvedRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="flex items-center gap-8">
      {[
        { title: 'Home', action: scrollToTop },
        { title: 'About Us', action: scrollToAbout },
        { title: 'Get Involved', action: () => navigate('/GetInvolved') }
      ].map((item) => (
        <motion.button
          key={item.title}
          onClick={item.action}
          className="relative group"
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          <span className="text-lg font-medium text-text-secondary group-hover:text-text-primary transition-all duration-300">
            {item.title}
          </span>
          <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-accent-blue group-hover:w-full transition-all duration-300" />
        </motion.button>
      ))}
    </div>
  );
};

export default HomePage; 