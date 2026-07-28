use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig, CudaSlice};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

pub struct GpuEvaluator {
    dev: Arc<CudaDevice>,
    func: CudaFunction,
    // Persistent buffers on the device (Phase 22.4 Optimization)
    d_bitsets:  Option<CudaSlice<u32>>,
    d_prices:   Option<CudaSlice<f32>>,
    d_atrs:     Option<CudaSlice<f32>>,
    d_btc_vols: Option<CudaSlice<f32>>,
}

impl GpuEvaluator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let dev = CudaDevice::new(0)?;
        
        // Embed CUDA kernel source (Phase 22.4 portability)
        let kernel_src = include_str!("gpu_kernels.cu");
        let ptx = compile_ptx(kernel_src)?;
        
        dev.load_ptx(ptx, "backtest_module", &["backtest_kernel"])?;
        let func = dev.get_func("backtest_module", "backtest_kernel").ok_or("Kernel not found")?;

        Ok(Self { 
            dev, 
            func,
            d_bitsets: None,
            d_prices: None,
            d_atrs: None,
            d_btc_vols: None,
        })
    }

    /// Upload constant data to GPU once per session.
    pub fn set_constants(
        &mut self,
        bitsets_raw: &[u32],
        prices: &[f32],
        atrs: &[f32],
        btc_vols: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.d_bitsets = Some(self.dev.htod_copy(bitsets_raw.to_vec())?);
        self.d_prices = Some(self.dev.htod_copy(prices.to_vec())?);
        self.d_atrs = Some(self.dev.htod_copy(atrs.to_vec())?);
        self.d_btc_vols = Some(self.dev.htod_copy(btc_vols.to_vec())?);
        Ok(())
    }

    pub fn evaluate_batch(
        &self,
        genome_params: &[f32],  // [N_GENOMES * 5]
        n_candles: i32,
        n_u32: i32,
        n_genomes: i32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let d_bitsets = self.d_bitsets.as_ref().ok_or("Constants not set")?;
        let d_prices = self.d_prices.as_ref().ok_or("Constants not set")?;
        let d_atrs = self.d_atrs.as_ref().ok_or("Constants not set")?;
        let d_btc_vols = self.d_btc_vols.as_ref().ok_or("Constants not set")?;

        // Transfer only dynamic params & allocate output
        let d_params = self.dev.htod_copy(genome_params.to_vec())?;
        let d_out = self.dev.alloc_zeros::<f32>(n_genomes as usize)?;

        let threads_per_block = 256;
        let blocks = (n_genomes as u32 + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.func.clone().launch(cfg, (
                d_bitsets,
                d_prices,
                d_atrs,
                d_btc_vols,
                &d_params,
                n_candles,
                n_u32,
                n_genomes,
                &d_out,
            ))?;
        }

        Ok(self.dev.dtoh_sync_copy(&d_out)?)
    }
}
