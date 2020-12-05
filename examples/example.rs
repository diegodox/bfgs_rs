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

    fn bfgs(&mut self) -> Result<(), String> {
        // initialize
        if !Self::params_is_valid(self.params()) {
            return Err(r#"init param has invalid value."#.to_string());
        }
        let (init_cost, init_grad) = self.calc_cost_and_grad(self.params());
        println!(
            "init param: {},init cost: {}, init grad: {}",
            self.params(),
            init_cost,
            init_grad
        );
        // best params
        let mut best_param = self.params().to_owned();
        let mut best_cost = init_cost;
        // internal values
        let mut current_cost = init_cost;
        let mut current_grad = init_grad;
        let mut inv_hessian = ndarray::Array2::<f64>::eye(Self::PARAM_DIM);
        let mut search_direction = {
            let neg_eye = ndarray::Array2::<f64>::eye(Self::PARAM_DIM).map(|e| e * -1f64);
            neg_eye.dot(&current_grad)
        };

        if bfgs::squared_l2_norm(search_direction.view()) < Self::TOL_SEARCH.powi(2) {
            self.set_params(best_param);
            return Ok(());
        }

        for _ in 0..Self::BFGS_MAX_ITER {
            self.set_params(
                match self.backtracking_line_search(
                    self.params(),
                    current_cost,
                    current_grad.view(),
                    search_direction.view(),
                ) {
                    Ok(v) => v,
                    Err(_) => return Err(r#"line search returned"#.to_string()),
                },
            );

            let (new_cost, new_grad) = self.calc_cost_and_grad(self.params());

            if new_cost < best_cost {
                best_param = self.params().to_owned();
                best_cost = new_cost;
            }

            if bfgs::squared_l2_norm(search_direction.view()) < Self::TOL_SEARCH.powi(2) {
                self.set_params(best_param);
                return Ok(());
            } else {
                println!("search is still: {}", search_direction.view())
            }

            if current_cost - new_cost < Self::TOL_COST {
                self.set_params(best_param);
                return Ok(());
            } else {
                println!("delta cost is: {}", current_cost - new_cost)
            }

            let delta_grad = new_grad.clone() - current_grad;
            current_grad = new_grad;
            current_cost = new_cost;

            let (new_search_direction, new_inv_hessian) = Self::update_search_direction(
                &search_direction.view(),
                &inv_hessian.view(),
                &current_grad.view(),
                &delta_grad.view(),
            );
            search_direction = new_search_direction;
            inv_hessian = new_inv_hessian;
        }
        println!("BFGS: UNREACHED TO BEST");
        self.set_params(best_param);
        Ok(())
    }

    fn backtracking_line_search(
        &self,
        params: ArrayView1<f64>,
        current_cost: f64,
        current_grad: ArrayView1<f64>,
        search_direction: ArrayView1<f64>,
    ) -> Result<Array1<f64>, ()> {
        let tol = -1f64 * Self::C * current_grad.t().dot(&search_direction);
        let mut alpha = Self::ALPHA_INIT;

        println!("line search");
        for i in 0..Self::LINE_SEARCH_MAX_ITER {
            println!("step: {}", i);
            let new_param = params.to_owned() + search_direction.map(|e| e * alpha);
            if !Self::params_is_valid(new_param.view()) {
                // パラメータが範囲外になってるからalpha小さく
                alpha *= Self::TAU;
                continue;
            }
            let new_cost = BFGS::calc_cost(self, new_param.view());
            if current_cost - new_cost > alpha * tol {
                // alpha確定
                return Ok(new_param);
            }
            // 行き過ぎてるからalpha小さく
            alpha *= Self::TAU;
        }
        Err(())
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
