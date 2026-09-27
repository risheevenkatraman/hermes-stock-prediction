import { Amplify } from "aws-amplify";
import { fetchAuthSession, signInWithRedirect, signOut } from "aws-amplify/auth";
import "aws-amplify/auth/enable-oauth-listener";

const userPoolId = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID;
const userPoolClientId = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID;
const domain = process.env.NEXT_PUBLIC_COGNITO_DOMAIN;
const appUrl = process.env.NEXT_PUBLIC_APP_URL;
export const authConfigured = Boolean(userPoolId && userPoolClientId && domain && appUrl);

if (authConfigured) {
  Amplify.configure({ Auth: { Cognito: {
    userPoolId: userPoolId!, userPoolClientId: userPoolClientId!,
    loginWith: { oauth: {
      domain: domain!, scopes: ["openid", "email", "profile"],
      redirectSignIn: [`${appUrl!.replace(/\/$/, "")}/`],
      redirectSignOut: [`${appUrl!.replace(/\/$/, "")}/`], responseType: "code",
    } },
  } } });
}

export async function session() {
  if (!authConfigured) return null;
  const result = await fetchAuthSession();
  if (!result.tokens) return null;
  return { token: result.tokens.accessToken.toString(), name: String(result.tokens.idToken?.payload.email ?? "Your account") };
}

export async function login(google = false) {
  if (!authConfigured) throw new Error("Account sign-in will be available when this workspace is connected to Cognito.");
  await signInWithRedirect(google ? { provider: "Google" } : undefined);
}
export { signOut };
