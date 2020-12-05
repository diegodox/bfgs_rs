# BFGS

Rust implementation of [BFGS algorithm](https://www.wikiwand.com/en/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).

Find parameters minimize (preferably twice continuously differentiable) function.

## How to use

- Add `Cargo.toml`
  ```toml
  [dependencies]
  bfgs = { git = "https://github.com/diegodox/bfgs_rs" }
  ```
- Impl Trait `BFGS` for your problem struct. You need to hold parameters on your struct.
- Call `bfgs` function that change your struct's parameter optimized.

## Example

See `examples/example.rs`.

```rust
use bfgs::prelude::*;
use ndarray::ArrayView1;
/// represent x^2 + y^2
struct Foo {
    x: f64,
    y: f64,
    params: Array1<f64>,
}
impl BFGS for Foo {
    const PARAM_DIM: usize = 2;

    const ALPHA_INIT: f64 = 1.0;

    const TAU: f64 = 0.5;

    const C: f64 = 1e-4;

    const TOL_SEARCH: f64 = 4E-8;

    const TOL_COST: f64 = 1E-10;

    const BFGS_MAX_ITER: usize = 10;

    const LINE_SEARCH_MAX_ITER: usize = 100;

    fn calc_cost(&self, params: ndarray::ArrayView1<f64>) -> f64 {
        (self.x - params[0]).powi(2) + (self.y - params[1]).powi(2)
    }

    fn calc_cost_and_grad(&self, params: ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) {
        let cost = self.calc_cost(params);
        let grad_x = 2f64 * params[0] - 2f64 * self.x;
        let grad_y = 2f64 * params[1] - 2f64 * self.y;
        (cost, arr1(&[grad_x, grad_y]))
    }

    fn params_is_valid(params: ndarray::ArrayView1<f64>) -> bool {
        params[0].is_normal() && params[1].is_normal()
    }

    fn params(&self) -> ArrayView1<f64> {
        self.params.view()
    }

    fn set_params(&mut self, params: Array1<f64>) {
        self.params = params
    }
}

fn main() {
    let mut target = Foo {
        x: 10f64,
        y: -7f64,
        params: arr1(&[3.5f64, 20f64]),
    };
    let _ = target.bfgs();
    println!("Analytical solution: [10,-7]");
    println!("Approximate solution: {}", target.params());
}
```
