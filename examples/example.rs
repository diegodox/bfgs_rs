use bfgs::prelude::*;
use ndarray::ArrayView1;
struct Foo {
    x: f64,
    y: f64,
    params: Array1<f64>,
}
impl Foo {
    fn calc_cost(&self, x: f64, y: f64) -> f64 {
        (self.x - x).powi(2) + (self.y - y).powi(2)
    }
    fn calc_grad_x(&self, x: f64) -> f64 {
        // x^2 - 2x self.x + self.x^2
        // 2x - 2 self.x
        2f64 * x - 2f64 * self.x
    }
    fn calc_grad_y(&self, y: f64) -> f64 {
        // x^2 - 2x self.x + self.x^2
        // 2x - 2 self.x
        2f64 * y - 2f64 * self.y
    }
    fn calc_cost_and_grad(&self, param: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let x = param[0];
        let y = param[1];
        (
            self.calc_cost(x, y),
            arr1(&[self.calc_grad_x(x), self.calc_grad_y(y)]),
        )
    }
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
        self.calc_cost(params[0], params[1])
    }

    fn calc_cost_and_grad(&self, params: ndarray::ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calc_cost_and_grad(params)
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
