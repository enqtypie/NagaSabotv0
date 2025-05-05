# NagaSabot Frontend Deployment on Render

This document provides instructions for deploying the NagaSabot frontend to Render.

## Prerequisites

- A Render account
- Your backend already deployed on Render

## Deployment Steps

1. **Create a new Static Site on Render**
   - Go to your Render dashboard
   - Click "New" and select "Static Site"
   - Connect your GitHub repository

2. **Configure the Static Site**
   - **Name**: `nagsabot-frontend` (or any name you prefer)
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave blank
   - **Runtime**: `Node`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist/naga-sabot/browser`

3. **Add Environment Variables**
   - Add the following environment variable:
     - **Key**: `FLASK_APP_API_URL`
     - **Value**: `https://nagasabot.onrender.com` (replace with your actual backend URL)

4. **Deploy**
   - Click "Create Static Site"
   - Render will build and deploy your site

## Troubleshooting

- If you encounter CORS issues, make sure your backend allows requests from your frontend domain
- If the build fails, check the build logs for errors
- If the site doesn't load correctly, check the browser console for errors

## Updating the Deployment

- Any changes pushed to your main branch will automatically trigger a new deployment
- You can also manually redeploy from the Render dashboard 