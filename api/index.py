from app.main import app

# Vercel expects the app to be available at module level
# This is the ASGI application that Vercel will use
handler = app
