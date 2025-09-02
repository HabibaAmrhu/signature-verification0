# 🚀 Deployment Guide

## Quick Deploy to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Your trained model file (`siamese_model.keras`)

### Step-by-Step Deployment

#### 1. Prepare Your Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy!"

#### 3. Your app will be live at:
`https://[your-app-name].streamlit.app`

### Important Notes
- **Model File**: Ensure `siamese_model.keras` is in your repo (should be ~50MB)
- **Dependencies**: All requirements are in `requirements.txt`
- **Memory**: Streamlit Cloud provides 1GB RAM (sufficient for this app)
- **Updates**: App auto-updates when you push to GitHub

## Alternative Deployment Options

### Local Development
```bash
# Clone and run locally
git clone [your-repo-url]
cd signature-verification-ai
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Heroku Deployment
1. Create `Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Common Issues

**Model file too large for GitHub:**
- Use Git LFS: `git lfs track "*.keras"`
- Or use cloud storage and download in app

**Memory issues:**
- Optimize model loading with `@st.cache_resource`
- Consider model quantization for smaller size

**Slow loading:**
- Model caching is already implemented
- Consider using a CDN for model files

**Dependencies not installing:**
- Check Python version compatibility
- Verify all packages in requirements.txt

### Performance Optimization

1. **Model Caching**: Already implemented with `@st.cache_resource`
2. **Image Processing**: Optimized preprocessing pipeline
3. **Batch Processing**: Efficient memory usage for multiple images
4. **Progress Indicators**: User feedback during processing

## Monitoring & Analytics

### Streamlit Cloud Analytics
- Built-in usage analytics
- Performance monitoring
- Error tracking

### Custom Analytics (Optional)
Add to your app:
```python
import streamlit as st

# Track usage
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
st.session_state.usage_count += 1
```

## Security Considerations

1. **No Data Storage**: Images are processed in memory only
2. **HTTPS**: Automatic with Streamlit Cloud
3. **No Authentication**: Public access (add auth if needed)
4. **Rate Limiting**: Consider implementing for production use

## Scaling

### For High Traffic
- Consider upgrading to Streamlit Cloud Pro
- Implement caching strategies
- Use load balancing for multiple instances

### Enterprise Deployment
- Deploy on AWS/GCP/Azure
- Use container orchestration (Kubernetes)
- Implement proper monitoring and logging

## Support

If you encounter issues:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the troubleshooting section above
3. Open an issue in the GitHub repository