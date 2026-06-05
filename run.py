import uvicorn
import sys
import logging
from backend.config import HOST, PORT
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Launcher")

if __name__ == "__main__":
    logger.info("Initializing Speech Schizophrenia Estimation Server...")
    logger.info(f"Access the Clinical Dashboard at: http://{HOST}:{PORT}/")
    
    try:
        

        uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",
    port=int(os.getenv("PORT", 8000)),
    reload=False
)
    except KeyboardInterrupt:
        logger.info("Server terminated by user request.")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
