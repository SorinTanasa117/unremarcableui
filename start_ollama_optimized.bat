@echo off
title Start Ollama Optimized for Intel Core Ultra 9 185H
echo =====================================================================
echo    Ollama Optimization Script for Intel Core Ultra 9 185H (32GB RAM)
echo =====================================================================
echo.
echo [1/5] Saving persistent Ollama environment variables (setx)...
setx OLLAMA_FLASH_ATTENTION 1 >nul
setx OLLAMA_KV_CACHE_TYPE q8_0 >nul
setx OLLAMA_NUM_PARALLEL 1 >nul
setx OLLAMA_MAX_LOADED_MODELS 1 >nul
setx OLLAMA_LLM_LIBRARY vulkan >nul
setx OLLAMA_KEEP_ALIVE -1 >nul

echo [2/5] Applying same variables to this shell for immediate serve...
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_LLM_LIBRARY=vulkan
set OLLAMA_KEEP_ALIVE=-1

echo [3/5] Thread tuning moved to UI toggle (6 or 8 threads per request).
echo [4/5] Vulkan backend requested (OLLAMA_LLM_LIBRARY=vulkan).
echo [5/5] Launching Ollama server...
echo.
echo Optimizations Active!
echo ---------------------------------------------------------------------
ollama serve
