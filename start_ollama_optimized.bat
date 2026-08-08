@echo off
title Start Ollama Optimized for Intel Core Ultra 9 185H
echo =====================================================================
echo    Ollama Optimization Script for Intel Core Ultra 9 185H (32GB RAM)
echo =====================================================================
echo.
echo [1/5] Configuring thread limit to 6 (matching physical P-Cores)...
echo       (Utilizing only the 6 Performance Cores avoids the E-core
echo       and hyperthreading memory bandwidth bottleneck)
set OLLAMA_NUM_THREADS=6

echo [2/5] Enabling Flash Attention...
set OLLAMA_FLASH_ATTENTION=1

echo [3/5] Setting KV Cache Type to q8_0 (reduces KV cache memory by 50%%)...
set OLLAMA_KV_CACHE_TYPE=q8_0

echo [4/5] Restricting loaded models and parallel processing to avoid swap...
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1

echo [5/5] Launching Ollama server...
echo.
echo Optimizations Active!
echo ---------------------------------------------------------------------
ollama serve
