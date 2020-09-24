pub mod prelude;
use ndarray::{arr1, Array1, Array2, ArrayView1};
/// ```
/// // Parameter count. if you have 3 parameters to optimize, this will be `3usize`.
/// const PARAM_DIM: usize;
/// // Backtracking_line_search_params
/// const ALPHA_INIT: f64 = 1f64;
/// // Backtracking_line_search_params
/// // 大きさの減少割合
/// const TAU: f64 = 0.8;
/// // Backtracking_line_search_params
/// // コスト変化量の最小値の係数
/// const C: f64 = 0.001;
/// // Minimum gradient norm to continue optimization
/// const TOL_GRAD: f64 = 4e-8;
/// // Minimum cost change to continue optimization
/// const TOL_COST: f64 = 4e-8;
/// // BFGS max iteration count
/// const BFGS_MAX_ITER: usize = 10;
/// // Line search max iteration count
/// const LINE_SEARCH_MAX_ITER: usize = 100;
/// ```
pub trait BfgsParams {
    /// Parameter count. if you have 3 parameters to optimize, this will be `3usize`.
    const PARAM_DIM: usize;
    /// Backtracking_line_search_params
    const ALPHA_INIT: f64;
    /// Backtracking_line_search_params
    /// 大きさの減少割合
    const TAU: f64;
    /// Backtracking_line_search_params
    /// コスト変化量の最小値の係数
    const C: f64;
    /// Minimum gradient norm to continue optimization
    const TOL_GRAD: f64;
    /// Minimum cost change to continue optimization
    const TOL_COST: f64;
    /// BFGS max iteration count
    const BFGS_MAX_ITER: usize = 10;
    /// Line search max iteration count
    const LINE_SEARCH_MAX_ITER: usize = 100;
}

/// ## BFGS Optimize function
///
/// RETURN: Optimized Parmeters
///
/// - `target`: target object to optimize.
/// - `init_param`: init value of param to optimize.
/// - `calc_cost_and_grad`: function calcrate cost and grad cost using `prams`.
/// - `param_check`: function return Err when `params` are invalid.
/// - `_bfgs_param`: trait `BfgsParams`.
pub fn bfgs<T, B: BfgsParams>(
    target: &T,
    init_param: &[f64],
    calc_cost_and_grad: fn(target: &T, params: &Array1<f64>) -> (f64, Array1<f64>),
    param_check: fn(target: &T, params: &Array1<f64>) -> Result<(), ()>,
    _bffgs_params: &B,
) -> Result<Array1<f64>, String> {
    let backtracking_line_search =
        |param: &Array1<f64>,
         current_cost: f64,
         current_grad: &Array1<f64>,
         search_direction: &Array1<f64>| {
            let t = -1f64 * B::C * current_grad.t().dot(search_direction);
            let mut alpha = B::ALPHA_INIT;
            for _ in 0..B::LINE_SEARCH_MAX_ITER {
                let new_param = param.clone() + search_direction.map(|e| e * alpha);
                if param_check(target, &new_param).is_err() {
                    // パラメータが範囲外になってるからalpha小さく
                    alpha *= B::TAU;
                    continue;
                }
                let (new_cost, _new_grad) = calc_cost_and_grad(target, &new_param);
                if current_cost - new_cost > alpha * t {
                    // alpha確定
                    break;
                }
                // 行き過ぎてるからalpha小さく
                alpha *= B::TAU;
            }
            let new_param = param.clone() + search_direction.map(|e| e * alpha);
            if param_check(target, &new_param).is_err() {
                return Err("parameters went invalid while backtracking line search.".to_string());
            }else{
                return Ok(alpha)
            }
        };
    // DFP Update
    let update_search_direction = |search_direction: &Array1<f64>,
                                   inv_hessian: Array2<f64>,
                                   grad: &Array1<f64>,
                                   delta_grad: &Array1<f64>| {
        let neg_eye = Array2::<f64>::eye(B::PARAM_DIM).map(|e| e * -1f64);
        let v = inv_hessian.dot(delta_grad);
        let c1 = 1f64 / delta_grad.t().dot(search_direction);
        let c2 = 1f64 + c1 * delta_grad.t().dot(&v);
        let mut new_inv_hessian =
            Array2::<f64>::from_shape_fn((B::PARAM_DIM, B::PARAM_DIM), |(i, j)| {
                if j >= i {
                    inv_hessian[[i, j]]
                        + c1 * ((c2 * search_direction[i] * search_direction[j])
                            - (v[i] * search_direction[j] + search_direction[i] * v[j]))
                } else {
                    0f64
                }
            });
        for i in 0..B::PARAM_DIM {
            for j in i..B::PARAM_DIM {
                new_inv_hessian[[j, i]] = new_inv_hessian[[i, j]];
            }
        }
        let new_search_direction: Array1<f64> = neg_eye.dot(&(inv_hessian.dot(grad)));
        (new_search_direction, new_inv_hessian)
    };
    fn l2_norm(x: ArrayView1<f64>) -> f64 {
        x.dot(&x).sqrt()
    }
    if init_param.len() != B::PARAM_DIM {
        return Err("init param has invalid length.".to_string());
    }
    let mut param = arr1(init_param);
    if param_check(target, &param).is_err() {
        return Err("init param has invalid value.".to_string());
    }
    let (init_cost, init_grad) = calc_cost_and_grad(target, &param);

    let mut best_param = param.clone();
    let mut best_cost = init_cost.clone();

    let mut current_cost = init_cost;
    let mut current_grad = init_grad;
    let mut inv_hessian = Array2::<f64>::eye(B::PARAM_DIM);
    let mut search_direction = {
        let neg_eye = Array2::<f64>::eye(B::PARAM_DIM).map(|e| e * -1f64);
        neg_eye.dot(&current_grad)
    };

    if l2_norm(search_direction.view()) < B::TOL_GRAD {
        // 探索方向の大きさが小さいので終了
        return Ok(best_param);
    }

    for _ in 0..B::BFGS_MAX_ITER {
        let alpha =
            backtracking_line_search(&param, current_cost, &current_grad, &search_direction)?;
        search_direction *= alpha;
        param = param.clone() + search_direction.clone();
        let (new_cost, new_grad) = calc_cost_and_grad(target, &param);
        if new_cost < best_cost {
            best_param = param.clone();
            best_cost = new_cost;
        }
        if l2_norm(search_direction.view()) < B::TOL_GRAD {
            // 探索方向の大きさが小さいので終了
            return Ok(best_param);
        }
        if current_cost - new_cost < B::TOL_COST {
            // コストの変化が小さいので終了
            return Ok(best_param);
        }
        let delta_grad = new_grad.clone() - current_grad;
        current_grad = new_grad;
        current_cost = new_cost;

        let (new_search_direction, new_inv_hessian) =
            update_search_direction(&search_direction, inv_hessian, &current_grad, &delta_grad);
        search_direction = new_search_direction;
        inv_hessian = new_inv_hessian;
    }
    Ok(best_param)
}
