use bfgs::prelude::*;
struct Foo {
    x: f64,
    y: f64,
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
    fn calc_cost_and_grad(&self, param: &Array1<f64>) -> (f64, Array1<f64>) {
        let x = param[0];
        let y = param[1];
        (
            self.calc_cost(x, y),
            arr1(&[self.calc_grad_x(x), self.calc_grad_y(y)]),
        )
    }
    fn param_check(&self, param: &Array1<f64>) -> Result<(), ()> {
        if param[0].is_normal() && param[1].is_normal() {
            Ok(())
        } else {
            Err(())
        }
    }
}
struct Param;
impl BfgsParams for Param {
    const PARAM_DIM: usize = 2;
    const ALPHA_INIT: f64 = 1f64;
    const TAU: f64 = 0.8;
    const TOL_COST: f64 = 4e-8;
    const C: f64 = 0.001;
    const GRAD_TOL: f64 = 4e-8;
}

fn main() {
    let foo = Foo { x: 10f64, y: -7f64 };
    let init_param = [1f64,20f64];
    let bfgs_param = Param;
    let result = bfgs(
        &foo,
        &init_param,
        Foo::calc_cost_and_grad,
        Foo::param_check,
        &bfgs_param,
    )
    .unwrap();
    println!("Analytical solution: [10,-7]");
    println!("Approximate solution: {}",result);
}
