pub mod prelude;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub trait BFGS {
    /// Parameter count. if you have 3 parameters to optimize, this will be `3usize.
    const PARAM_DIM: usize;
    /// Backtracking_line_search_params, default value is 1.0
    const ALPHA_INIT: f64 = 1.0;
    /// Backtracking_line_search_params, default value is 0.5
    /// 線探索時の減少割合
    const TAU: f64 = 0.5;
    /// Backtracking_line_search_params
    /// 線探索時のコスト変化量の最小値の係数
    const C: f64;
    /// Minimum gradient norm to continue optimization, default value is 4E-8
    const TOL_GRAD: f64 = 1E-5;
    /// Minimum search direction norm to continue optimization, default value is 4E-8
    const TOL_SEARCH: f64 = 4E-8;
    /// Minimum cost change to continue optimization, default value is 1E-10
    const TOL_COST: f64 = 1E-10;
    /// BFGS max iteration count, default value is 10
    const BFGS_MAX_ITER: usize = 10;
    /// Line search max iteration count, defalut value is 100
    const LINE_SEARCH_MAX_ITER: usize = 100;

    fn calc_cost(&self, params: ArrayView1<f64>) -> f64;
    fn calc_cost_and_grad(&self, params: ArrayView1<f64>) -> (f64, Array1<f64>);
    fn params_is_valid(params: ArrayView1<f64>) -> bool;

    fn bfgs(&mut self, init: Array1<f64>) -> Result<Array1<f64>, String> {
        // initialize
        if !Self::params_is_valid(init.view()) {
            return Err(r#"init param has invalid value."#.to_string());
        }
        let (init_cost, init_grad) = self.calc_cost_and_grad(init.view());
        // best params
        let mut best_param = init.to_owned();
        let mut best_cost = init_cost;
        // internal values
        let mut current_param = init;
        let mut current_cost = init_cost;
        let mut current_grad = init_grad;
        let mut inv_hessian = Array2::<f64>::eye(Self::PARAM_DIM);
        let mut search_direction = {
            let neg_eye = Array2::<f64>::eye(Self::PARAM_DIM).map(|e| e * -1f64);
            neg_eye.dot(&current_grad)
        };

        if squared_l2_norm(search_direction.view()) < Self::TOL_SEARCH.powi(2) {
            return Ok(current_param);
        }

        for _ in 0..Self::BFGS_MAX_ITER {
            current_param = match self.backtracking_line_search(
                current_param.view(),
                current_cost,
                current_grad.view(),
                search_direction.view(),
            ) {
                Ok(v) => v,
                Err(_) => return Err(r#"line search returned"#.to_string()),
            };

            let (delta_cost, delta_grad) = {
                let (new_cost, new_grad) = self.calc_cost_and_grad(current_param.view());

                if new_cost < best_cost {
                    best_param = current_param.to_owned();
                    best_cost = new_cost;
                }

                let delta_cost = current_cost - new_cost;
                let delta_grad = new_grad.clone() - current_grad;

                current_grad = new_grad;
                current_cost = new_cost;

                (delta_cost, delta_grad)
            };

            if delta_cost < Self::TOL_COST {
                return Ok(best_param);
            }

            if squared_l2_norm(current_grad.view()) < Self::TOL_GRAD.powi(2) {
                return Ok(best_param);
            }

            let (new_search_direction, new_inv_hessian) = Self::update_search_direction(
                &search_direction.view(),
                &inv_hessian.view(),
                &current_grad.view(),
                &delta_grad.view(),
            );
            search_direction = new_search_direction;
            inv_hessian = new_inv_hessian;

            if squared_l2_norm(search_direction.view()) < Self::TOL_SEARCH.powi(2) {
                return Ok(best_param);
            }
        }
        println!("BFGS: UNREACHED TO BEST");
        Ok(best_param)
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

        for _ in 0..Self::LINE_SEARCH_MAX_ITER {
            let new_param = params.to_owned() + search_direction.map(|e| e * alpha);
            if !Self::params_is_valid(new_param.view()) {
                // パラメータが範囲外になってるからalpha小さく
                alpha *= Self::TAU;
                continue;
            }
            let new_cost = self.calc_cost(new_param.view());
            if current_cost - new_cost > alpha * tol {
                // alpha確定
                return Ok(new_param);
            }
            // 行き過ぎてるからalpha小さく
            alpha *= Self::TAU;
        }
        Err(())
    }
    fn update_search_direction(
        search_direction: &ArrayView1<f64>,
        inv_hessian: &ArrayView2<f64>,
        grad: &ArrayView1<f64>,
        delta_grad: &ArrayView1<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let neg_eye = Array2::<f64>::eye(Self::PARAM_DIM).map(|e| e * -1f64);
        let v = inv_hessian.dot(delta_grad);
        let c1 = 1f64 / delta_grad.t().dot(search_direction);
        let c2 = 1f64 + c1 * delta_grad.t().dot(&v);
        let mut new_inv_hessian =
            Array2::<f64>::from_shape_fn((Self::PARAM_DIM, Self::PARAM_DIM), |(i, j)| {
                if j >= i {
                    inv_hessian[[i, j]]
                        + c1 * ((c2 * search_direction[i] * search_direction[j])
                            - (v[i] * search_direction[j] + search_direction[i] * v[j]))
                } else {
                    0f64
                }
            });
        for i in 0..Self::PARAM_DIM {
            for j in i..Self::PARAM_DIM {
                new_inv_hessian[[j, i]] = new_inv_hessian[[i, j]];
            }
        }
        let new_search_direction: Array1<f64> = neg_eye.dot(&(inv_hessian.dot(grad)));
        (new_search_direction, new_inv_hessian)
    }
}

pub fn squared_l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x)
}
